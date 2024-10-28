import random
from torch.backends import cudnn
from dataset.galaxy_dataset import *
from args import *
import click
import dnnlib
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from utils import schemas
from utils.tool import *
from models.model_utils import *
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch
import numpy as np
import gc
import json
import torch.nn.functional as F
from scipy.stats import dirichlet
import warnings
warnings.filterwarnings("ignore")
root = "./"
import csv

class SphericalKMeans:
    def __init__(self, n_clusters, init_centers, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters # num of cluster center, equal to num of answers
        self.max_iters = max_iters # maximum iteration for Kmeans
        self.tol = tol # tolerence between last center and new center
        self.centroids = init_centers # initial centroids, equal to weight of W_s
        self.question_answer_pairs = gz2_pairs # galaxy zoo answer pairs, from W21
        self.dependencies = gz2_and_decals_dependencies # galaxy zoo answer dependicies, from W21
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies) # galaxy zoo answer schema, from W21
    def threshold_and_modify(self, tensor):
        return tensor # extension for assign distance threshold between centroids and feature vectors
    def fit(self, X):
        for iter in range(self.max_iters):
            prev_centroids = self.centroids.clone() # last centroids
            similarities = (torch.matmul(X, torch.Tensor(prev_centroids.T).to(X.device))) # compute cosine similarity
            similarities = self.threshold_and_modify(similarities.clone()) # threshold and modify similarities
            cluster_assignments = torch.zeros_like(similarities) # initialize cluster assignments
            for q_n in range(len(self.schema.question_index_groups)): # iter among questions
                q_indices = self.schema.question_index_groups[q_n] 
                q_start = q_indices[0]
                q_end = q_indices[1]
                input_slice = similarities[:, q_start:q_end + 1] # get the slice of similarities for the current question
                max_prob, max_index = torch.max(input_slice, dim=1) # get the maximum similarity and its index
                condition = max_prob > -1
                cluster_assignments[condition, q_start + max_index[condition]] = 1
            if len(X) == 1:
                cluster_assignments = cluster_assignments.squeeze(0)
            for i in range(self.n_clusters):
                cluster_points = X[cluster_assignments[:,i] == 1]
                if cluster_points.size(0) > 0:
                    new_centroid = torch.mean(cluster_points, dim=0)
                    self.centroids[i] = new_centroid / new_centroid.norm(p=2)  # L2 normalize
            centroid_shift = torch.sum((self.centroids - prev_centroids) ** 2)
            if centroid_shift <= self.tol:
                break
            
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
def get_center(model):
    classifier_weights = model.module.net.classifier[1].weight.data
    classifier_weights = F.normalize(classifier_weights, p=2, dim=1)
    return classifier_weights # L2 normalize the weight of classifier

class DomainAdaptation():
    def __init__(self, tau, model, config, optimizer,scheduler):
        self.tau = tau # temperature hyper-parameter
        self.model = model # AI model
        self.config = config
        self.optimizer = optimizer
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
        self.device = "cuda:1" # cuda idx
        self.prototypes = None
        self.cluster_centers = None
        self.scheduler = scheduler
    def predict(self, run_dir, model, data_loader, num_samples, output_file, schema): # not important, related to your downstream tasks
        with open(run_dir+output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = []
            for answer in schema.answers:
                # meanf = mean votes, meanf = mean fraction, fraction = raw GZD-5 fraction
                header.extend([f"{answer.text}_meanv", f"{answer.text}_meanf", f"{answer.text}_label", f"{answer.text}_fraction"])
            for question in schema.questions:
                header.extend([f"{question.text}_entropy"])
            writer.writerow(header)
            model.eval()
            enable_dropout(model)
            for img, label in data_loader:
                img = img.to(self.device)
                expected_alpha = []
                expected_probs = []
                for mc_run in range(num_samples):
                    with torch.no_grad():
                        alpha = model(img)
                        expected_alpha.append(alpha)
                        expected_probs.append(vote2prob(alpha, schema.question_index_groups))
                mean_probs = torch.mean(torch.stack(expected_probs), dim=0)
                fraction = calculate_fraction(label, schema.question_index_groups)
                mean_alpha = torch.mean(torch.stack(expected_alpha), dim=0)
                entropy = torch.zeros(img.shape[0], 10)
                for q_n in range(len(schema.question_index_groups)):
                    q_indices = schema.question_index_groups[q_n]
                    q_start = q_indices[0]
                    q_end = q_indices[1]
                    for id in range(img.shape[0]):
                        alpha_q_id = mean_alpha[id][q_start:q_end+1].detach().cpu().numpy()
                        entropy[id,q_n] = dirichlet(alpha_q_id).entropy()
                for i in range(img.shape[0]):
                    row = []  # add image path
                    for answer in schema.answers:
                        row.extend([mean_alpha[i][answer.index].item(), mean_probs[i][answer.index].item(), label[i][answer.index].item(),fraction[i][answer.index].item()])
                    for q_n in range(len(schema.question_index_groups)):
                        row.append(entropy[i, q_n].item())
                    writer.writerow(row)
                    
    def compute_infoNCE_loss(self, z_t, W_s, pseudo_labels, t): # loss function
        total = torch.mm(z_t, W_s.t()) / t
        loss = torch.zeros((total.shape[0], 10))
        for q_n in range(len(self.schema.question_index_groups)):
            q_indices = self.schema.question_index_groups[q_n]
            q_start = q_indices[0]
            q_end = q_indices[1]
            loss_ = torch.log_softmax(total[:, q_start:q_end + 1], dim=1) * pseudo_labels[:, q_start:q_end + 1]
            loss[:, q_n] = loss_.sum(dim=1)
        return -loss.sum(dim=1).mean()
    
    def extract_feas(self, loader):
        """
        get feature vector and normalize, from batches
        """
        features_list = []
        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(loader, dynamic_ncols=True, desc="Extracting features"):
                X = X.to(self.device)
                features = self.model(X, get_features=True)
                features_list.append(features)
        features_matrix = torch.cat(features_list, dim=0)
        features_matrix = F.normalize(features_matrix, p=2, dim=1)
        return features_matrix
    
    def clustering(self, features_matrix, init_centers):
        """
        fitting k-means
        """
        skmeans = SphericalKMeans(n_clusters=34, init_centers=init_centers)
        skmeans.fit(features_matrix)
        cluster_centers =skmeans.centroids
        return features_matrix, torch.Tensor(cluster_centers).to(self.device)
    
    def initialize_centers(self, train_loader, init_centers, epoch):
        self.model.eval()
        features_matrix = self.extract_feas(train_loader)
        return self.clustering(features_matrix, init_centers)
    
    def pseudo_label(self, X, epoch):
        X_norm = F.normalize(X, p=2, dim=1)
        similarities = (torch.matmul(X_norm, self.cluster_centers.t().to(X_norm.device))) 
        percent = 0.1
         # only do top_10_percent distance which is confident, using percent instead of fixed value benefits high-dim latent space
        top_10_percent = int(percent * similarities.size(0))
        selected_rows = torch.zeros(similarities.size(0), dtype=torch.bool)
        for i in range(similarities.size(1)):
            _, indices = torch.topk(similarities[:, i], top_10_percent, largest=True)
            selected_rows[indices] = True
        similarities[~selected_rows,:] = 0
        result = torch.zeros_like(similarities)
        for q_n in range(len(self.schema.question_index_groups)):
            q_indices = self.schema.question_index_groups[q_n]
            q_start = q_indices[0]
            q_end = q_indices[1]
            input_slice = similarities[:, q_start:q_end + 1]
            max_prob, max_index = torch.max(input_slice, dim=1)
            condition = max_prob > 0.9 - epoch * 0.005 # pseudo label applying threshold
            result[condition, q_start + max_index[condition]] = 1
        return result
    
    def train_unsupervised_epoch(self, train_loader, epoch, writer):
        train_loss = 0
        self.model.train()
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for X, y_hat in tqdmDataLoader:
                self.model.eval()
                X, y_hat = X.to(self.device), y_hat.to(self.device)
                self.optimizer.zero_grad()
                features = self.model(X, get_features=True) # z_t^i
                features = features.to(self.device)
                z = F.normalize(features, p=2, dim=1) # L2 normalize
                self.model.train()
                loss_value = self.compute_infoNCE_loss(z, self.prototypes.to(self.device), y_hat, self.tau)
                loss_value.backward()
                self.optimizer.step()
                train_loss += loss_value.item()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epoch + 1,
                        "loss: ": loss_value.item(),
                        "LR": self.optimizer.param_groups[0]['lr'],
                    }
                )
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Unsupervised Training loss by steps', avg_train_loss, epoch)
        return avg_train_loss
    
    def validate(self, valid_loader, writer, epoch):
        valid_loss = 0
        self.model.eval()
        for X, y_hat in valid_loader:
            X = X.to(self.device)
            y_hat = y_hat.to(self.device)
            enable_dropout(self.model)
            with torch.no_grad():
                features = self.model(X, get_features=True)
                features = features.to(self.device)
                z = F.normalize(features, p=2, dim=1)
                loss_value = self.compute_infoNCE_loss(z, self.prototypes.to(self.device), y_hat, self.tau)
                valid_loss += loss_value.item()
        avg_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar('Unsupervised Validation loss by steps', avg_valid_loss, epoch)
        return avg_valid_loss
    
    def train_unsupervised(self, train_data, valid_data, init_loader):
        os.makedirs(self.config.save_dir + "log/", exist_ok=True)
        writer = SummaryWriter(self.config.save_dir + "log/")
        self.prototypes = get_center(self.model) # prototypes = source model's last layer (classifier, after flattened) weights
        self.cluster_centers = get_center(self.model) # it will be modified after k-means
        for param in self.model.module.net.classifier.parameters(): # freeze the classifier
            param.requires_grad = False
        for epoch in range(self.config.epochs):
            if epoch == 0:
                print(f"{self.config.save_dir.split('/')[-2]}_{epoch}.csv")
            features, self.cluster_centers = self.initialize_centers(init_loader, self.prototypes, epoch)  # initialize cluster centers with prototypes 
            
            # assigning pseudo label and initialize Dataloader for non-zero pseudo labels (not meet threshold)
            pseudo_labels = self.pseudo_label(features, epoch)
            train_data.set_pseudo_labels(pseudo_labels.detach().cpu())
            pseudo_indices = torch.nonzero((pseudo_labels.sum(dim=1)!=0)).squeeze().tolist()
            sub_train_data = Subset(train_data, pseudo_indices)
            train_loader  = DataLoader(dataset=sub_train_data, batch_size=self.config.batch_size,
                              shuffle=True, num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
            
            train_loss = self.train_unsupervised_epoch(train_loader, epoch, writer)
            
            self.save_checkpoint(epoch)
            print(f"epoch: {epoch}, train_loss: {train_loss}")
            valid_loader = DataLoader(dataset=valid_data, batch_size=self.config.batch_size,
                              shuffle=False, num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
            del features
            features = self.extract_feas(valid_loader)
            pseudo_labels = self.pseudo_label(features, epoch, False)
            valid_data.set_pseudo_labels(pseudo_labels.detach().cpu())
            pseudo_indices = torch.nonzero((pseudo_labels.sum(dim=1)!=0)).squeeze().tolist()
            sub_valid_data = Subset(valid_data, pseudo_indices)
            valid_loader = DataLoader(dataset=sub_valid_data, batch_size=self.config.batch_size,
                              shuffle=False, num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
            valid_loss = self.validate(valid_loader, writer, epoch)
            print(f"epoch: {epoch}, valid_loss: {valid_loss}")
            if self.scheduler is not None:
                self.scheduler.step()
            del train_loader
            gc.collect()
            print("------------start testing--------------")
            test_data = TestDataset(annotations_file="./dataset/overlap_north_raw.txt",
                                    transform=transforms.Compose([transforms.ToTensor()]), )
            test_loader = DataLoader(dataset=test_data, batch_size=256,
                                    shuffle=False, num_workers=8, pin_memory=True)
            self.predict(self.model, test_loader, num_samples=25, output_file=f"{self.config.save_dir.split('/')[-2]}_{epoch}.csv", schema=self.schema)
    def save_checkpoint(self, epoch):
        os.makedirs(f'{self.config.save_dir}/checkpoint', exist_ok=True)
        torch.save(self.model, f'{self.config.save_dir}/model_{epoch}.pt')

def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

@click.command()
@click.option('--save_dir',     help='Where to save the results', metavar='DIR',)
@click.option('--train_file',   help='train set',                 metavar='DIR',)
@click.option('--valid_file',   help='validation set',            metavar='DIR',)
@click.option('--test_file',    help='test set',                  metavar='DIR',)
@click.option('--epochs',       help='epochs',                    metavar='DIR',                default=100)
@click.option('--batch',        help='batch',                     metavar='INT',                type=click.IntRange(min=1), default=320)
@click.option('--patience',     help='patience',                  metavar='INT',                type=click.IntRange(min=1), default=15)
@click.option('--dropout_rate', help='dropout_rate',              metavar='INT',                type=click.IntRange(max=1), default=0.2)
@click.option('--weight_decay', help='weight_decay',              metavar='INT',                type=click.IntRange(max=1), default=1e-3)
@click.option('--tau',          help='tau hyperparameter',        metavar='INT',                default=0.05)
@click.option('--model_path',   help='model_path',                metavar='STR',                default="/data/public/renhaoye/mgs/ye_2023_sigmoid_raw/model_54.pt")

# Optional features.
@click.option('--resume',       help='Resume from given network pickle',   metavar='[PATH|URL]',type=str)
# Misc hyperparameters.
@click.option('--max_lr',       help='max learning rate  [default: varies]', metavar='FLOAT',   type=click.FloatRange(min=0), default=0.001)
@click.option('--lr',           help='base learning rate', metavar='FLOAT',                     type=click.FloatRange(min=0), default=0.001, show_default=True)
@click.option('--sample',       help='total sample nums', metavar='FLOAT',                      type=click.IntRange(min=1),   default=1)

# Misc settings.
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=3407, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--div_factor',   help='Loss scheduler division factor', metavar='INT',           type=click.IntRange(min=1), default=100000, show_default=True)
@click.option('--pct_start',    metavar='FLOAT',                                                type=click.FloatRange(min=0), default=0.3)
@click.option('--gpu',          help='GPU list to use', metavar='LIST',                         type=list, default=[1])

@click.option('--pin_memory',   metavar='BOOL',                                                 type=bool, default=True, show_default=True)
@click.option('--shuffle',      metavar='BOOL',                                                 type=bool, default=True, show_default=True)

@click.option('--subset',       help='subset', metavar='INT',                                   type=click.IntRange(min=0), show_default=True)
@click.option('--pytorch2_acc', help='pytorch2_acc', metavar='BOOL',                            type=bool, default=True, show_default=True)
@click.option('--raw_fits',     help='use raw_fits?', metavar='BOOL',                           type=bool, default=True, show_default=True)
@click.option('--gray_scale',   help='averge over channels?', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--png',          help='use png?', metavar='BOOL',                                type=bool, default=False, show_default=True)

def main(**kwargs):
    c = dnnlib.EasyDict({
        'transfer': transforms.Compose([
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        'betas': (0.9, 0.999),
    })
    opts = dnnlib.EasyDict(kwargs)
    c.lr = opts.lr
    c.tau = opts.tau
    c.epochs = opts.epochs
    c.max_lr = opts.max_lr
    c.div_factor = opts.div_factor
    c.weight_decay = opts.weight_decay
    c.pct_start=opts.pct_start
    c.dropout_rate = opts.dropout_rate
    c.save_dir = opts.save_dir
    c.train_file = opts.train_file
    c.valid_file = opts.valid_file
    c.test_file = opts.test_file
    c.batch = opts.batch
    c.workers = opts.workers
    c.fp32 = opts.fp32
    c.seed = opts.seed
    c.sample = opts.sample
    c.gpu = opts.gpu
    c.pytorch2_acc = opts.pytorch2_acc
    c.subset = opts.subset
    c.pin_memory = opts.pin_memory
    c.shuffle = opts.shuffle
    c.gray_scale = opts.gray_scale
    c.png = opts.png
    c.patience = opts.patience
    c.raw_fits = opts.raw_fits
    c.model_path = opts.model_path    
    init_rand_seed(c.seed)
    if c.fp32:
        torch.set_float32_matmul_precision('highest')
    else:
        torch.set_float32_matmul_precision('high')

    train_data = GalaxyDataset(annotations_file=c.train_file, transform=c.transfer)
    init_loader = DataLoader(dataset=train_data, batch_size=512,shuffle=c.shuffle, num_workers=c.workers, pin_memory=c.pin_memory, drop_last=True)
    valid_data = GalaxyDataset(annotations_file=c.valid_file, transform=c.transfer)
    if len(c.gpu) > 1:
        c.device = "cuda:0"
    else:
        c.device = f"cuda:{int(c.gpu[0])}"
    model = torch.load(c.model_path, map_location=c.device)
    model = torch.nn.DataParallel(model, device_ids=c.gpu)
    if isinstance(model, torch.nn.DataParallel):
        model_parameters = model.module.parameters()
    else:
        model_parameters = model.parameters() 
    c.opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.AdamW", lr=c.lr, weight_decay=c.weight_decay, betas=c.betas)
    c.opt_kwargs.params = model_parameters
    optimizer = dnnlib.util.construct_class_by_name(**c.opt_kwargs)
    
    c.sch_kwargs = dnnlib.EasyDict(class_name="torch.optim.lr_scheduler.OneCycleLR", optimizer=optimizer, max_lr=c.max_lr,\
        epochs=c.epochs, steps_per_epoch = len(train_data),div_factor=c.div_factor,pct_start=c.pct_start)
    scheduler = dnnlib.util.construct_class_by_name(**c.sch_kwargs)
    
    os.makedirs(c.save_dir, exist_ok=True)
    os.system(f"cp {os.path.abspath(__file__)} {c.save_dir}")
    
    adapter = DomainAdaptation(c.tau, model, c, optimizer, scheduler=scheduler)
    adapter.train_unsupervised(train_data, valid_data, init_loader)
    
if __name__ == "__main__":
    main()
