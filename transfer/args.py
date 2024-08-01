import torchvision.transforms as transforms

def get_data_config():
    return {
        'root': "./",
        'save_dir': "./ye_2023_uda_g/",
        'train_file': "./file_path.txt", # train file, each line is a sample, the first element is the path of the image
        'epochs': 200,
        'batch_size': 640,
        'patience': 5,
        'dropout_rate': 0.3,
        'WORKERS': 64,
        'transfer': transforms.Compose([
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        'lr': 1e-6,
        'betas': (0.9, 0.999),
        'phase': "training",
        'sample': 1,
        'dist_threshold': 0.,
        'model': "ye_2023_sigmoid_raw/model_54.pt",
        'tau': 0.05,
        'tmax': 365*2,
        'optimizer': "AdamW",
    }
