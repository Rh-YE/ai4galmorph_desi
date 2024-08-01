import torchvision.transforms as transforms


def get_data_config():
    return {
        'save_dir': "./ye_2023_sigmoid/", # log dir
        'train_file': "./train_1009.txt", # train file, each line is a sample, the first element is the path of the image, the rest are the votes
        'valid_file': "./valid_1009.txt",
        'epochs': 100,
        'batch_size': 128,
        'patience': 10,
        'dropout_rate': 0.3,
        'WORKERS': 128,
        'transfer': transforms.Compose([
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'phase': "training",
        'sample': 1,
        "T_max": 100
    }