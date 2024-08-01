import random
from torch.backends import cudnn
from dataset.galaxy_dataset import *
from args import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from utils import schemas
from training import losses
import argparse
from utils.tool import *
from torchvision.models import *
from models.mgs import MGSModel
from models.model_utils import *
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Trainer:
    def __init__(self, model, optimizer, config, iters):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = "cuda"
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
        self.early_stopping = EarlyStopping(patience=self.config.patience, delta=0.001, verbose=True)
        self.scheduler= OneCycleLR(optimizer, max_lr=self.config.lr, epochs=self.config.epochs, steps_per_epoch=iters,div_factor=100000)

    def dirichlet_loss_func(self, preds, labels):
        return losses.calculate_multiquestion_loss(labels, preds, self.schema.question_index_groups)

    def train_epoch(self, train_loader, epoch, writer):
        train_loss = 0
        question_loss = np.zeros(10)
        self.model.train()
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for X, label in tqdmDataLoader:
                label = torch.as_tensor(label, dtype=torch.long).to(self.device)
                X = X.to(self.device)
                output, _ = self.model(X)
                # output = softmax_output(output, self.schema.question_index_groups) * 99 + 1
                dirichlet_loss = torch.mean(self.dirichlet_loss_func(output, label), dim=0)
                loss_value = torch.sum(dirichlet_loss)
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()

                train_loss += torch.sum(dirichlet_loss).item()
                q_loss = dirichlet_loss.detach().cpu().numpy()
                question_loss += dirichlet_loss.detach().cpu().numpy()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epoch,
                        "loss: ": torch.sum(dirichlet_loss).item(),
                        "LR": self.optimizer.param_groups[0]['lr'],
                    }
                )
                self.scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Training loss by steps', avg_train_loss, epoch)
        return avg_train_loss

    def evaluate(self, valid_loader, epoch, writer):
        eval_loss = 0
        question_loss = np.zeros(10)
        with torch.no_grad():
            self.model.eval()
            for X, label in valid_loader:
                label = torch.as_tensor(label, dtype=torch.long).to(self.device)
                X = X.to(self.device)
                output, _ = self.model(X)
                dirichlet_loss = torch.mean(self.dirichlet_loss_func(output, label), dim=0)
                question_loss += dirichlet_loss.detach().cpu().numpy()
                eval_loss += torch.sum(dirichlet_loss).item()
        avg_eval_loss = eval_loss / len(valid_loader)
        writer.add_scalar('Validating loss by steps', avg_eval_loss, epoch)
        return avg_eval_loss

    def save_checkpoint(self, epoch):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch
        }
        os.makedirs(f'{self.config.save_dir}/checkpoint', exist_ok=True)
        torch.save(checkpoint, f'{self.config.save_dir}/checkpoint/ckpt_best_{epoch}.pth')
        torch.save(self.model.module, f'{self.config.save_dir}/model_{epoch}.pt')

    def train(self, train_loader, valid_loader):
        os.makedirs(self.config.save_dir + "log/", exist_ok=True)
        writer = SummaryWriter(self.config.save_dir + "log/")
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, epoch, writer)
            print(f"epoch: {epoch}, loss: {train_loss}")
            eval_loss = self.evaluate(valid_loader, epoch, writer)
            print(f"valid_loss: {eval_loss}")
            self.save_checkpoint(epoch)
            self.early_stopping(eval_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break


def main(config):
    model = MGSModel(dropout_rate=config.dropout_rate)
    device_ids = [0,1]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to("cuda:0")
    model = torch.compile(model)
    train_data = GalaxyDataset(annotations_file=config.train_file, transform=config.transfer)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    valid_data = GalaxyDataset(annotations_file=config.valid_file,
                               transform=transforms.Compose([transforms.ToTensor()]), )
    valid_loader = DataLoader(dataset=valid_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.WORKERS, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas)
    trainer = Trainer(model=model, optimizer=optimizer, config=config, iters=len(train_loader))
    trainer.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == "__main__":
    init_rand_seed(1926)
    data_config = get_data_config()
    info = get_data_config()  # change this line
    os.makedirs(info['save_dir'], exist_ok=True)
    with open(info['save_dir'] + "info.txt", "w") as w:  # change this line
        for each in info.keys():
            attr_name = each
            attr_value = info[each]  # change this line
            w.write(str(attr_name) + ':' + str(attr_value) + "\n")
    parser = argparse.ArgumentParser(description='MGS: Galaxy Classification Training')
    parser.add_argument('--save_dir', type=str, default=data_config['save_dir'],
                        help='Directory to save logs, checkpoints, and trained models')
    parser.add_argument('--train_file', type=str, default=data_config['train_file'],
                        help='Path to the training data annotations file')
    parser.add_argument('--valid_file', type=str, default=data_config['valid_file'],
                        help='Path to the validation data annotations file')

    parser.add_argument('--epochs', type=int, default=data_config['epochs'],
                        help='Number of epochs to training')
    parser.add_argument('--batch_size', type=int, default=data_config['batch_size'],
                        help='Batch size for training and validation')
    parser.add_argument('--WORKERS', type=int, default=data_config['WORKERS'],
                        help='Number of workers for data loading')
    parser.add_argument('--betas', type=tuple, default=data_config['betas'],
                        help='Optimizer parameters')
    parser.add_argument('--transfer', type=callable, default=data_config['transfer'],
                        help='Transforms to apply to the input data')
    parser.add_argument('--lr', type=float, default=data_config['lr'],
                        help='Learning rate for training')
    parser.add_argument('--patience', type=int, default=data_config['patience'],
                        help='Patience for early stopping')
    parser.add_argument('--phase', type=str, default=data_config['phase'],
                        help='Phase for training')
    parser.add_argument('--sample', type=int, default=data_config['sample'],
                        help='Sample nums for training')
    parser.add_argument('--dropout_rate', type=float, default=data_config['dropout_rate'],
                        help='Dropout rate for training')
    args = parser.parse_args()
    main(args)
