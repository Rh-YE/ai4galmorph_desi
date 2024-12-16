# From Galaxy Zoo DECaLS to BASS/MzLS
This repository contains the code for fine-tune a source domain model (trained on the DECaLS survey) to the BASS/MzLS surveys through unsupervised domain adaptation (UDA). For detailed information on the algorithm, please refer to the original paper (citation to be added).

## Overview
This repository represents work initiated by the first author of the paper since 2022, focused on galaxy morphology classification within the DESI Legacy Imaging Surveys (DESI-LIS). Throughout multiple iterations, this project evolved independently, and by the time the Galaxy Zoo DESI preprint was still unpublished, we had not communicated with Mike Walmsley or other contributors to the GZ DESI project. As we release this code, we recognize that the catalogue we present is not the first of its kind. Our aim is to complement the `Zoobot` series of models developed by Mike Walmsley et al. with an unsupervised domain adaptation approach.

The primary purpose of making this code open-source is to help readers interested in galaxy morphology classification understand the implementation of unsupervised domain adaptation algorithms and utilize our model weights for predictions. Given the rapid evolution of neural network algorithms, training models on DESI-LIS for galaxy morphology classification using `Zoobot` alone is relatively straightforward, with many models capable of achieving comparable performance. Therefore, this repository does not delve deeply into the source domain training methods.

Some portions of the code in this repository are based on the [`Zoobot`](https://joss.theoj.org/papers/10.21105/joss.05312) repository, particularly the label processing parts. 

## Structure

- `dataset/`: 
  - `galaxy_dataset.py`: Defining PyTorch Dataset class.
- `dnnlib/`: 
  - `util.py`: Includes methods for managing configuration files, inspired by the StyleGAN3 repo.
- `mgs/`: 
  - `args.py`: Training cofiguration.
  - `train.py`: Training code.
- `models/`:
  - `data_parallel.py`: multi-GPU training setups
  - `mgs.py`: model definition
  - `model_utils.py`: early stopping strategies
- `training/`: 
  - `losses.py`: `Zoobot` series losses' definition
- `paper/`: 
  - `inference.py`: Inference code, which involves reading models, fixing random seeds, and correctly converting prediction nodes into labels.
- `transfer/`: Implements unsupervised domain adaptation algorithms, with details in `cdcl.py`.
  - `args.py`: Unsuperviesd Domain Adaptation cofiguration.
  - `cdcl.py`: Unsuperviesd Domain Adaptation Algorithm and training code.
- `utils/`: Primarily includes label processing code used in `Zoobot`.


## Training and Inference
The source domain model training in this repository is entirely manual and does not utilize the `pytorch_lightning` package, which is now more commonly adopted as in `Zoobot`. This choice aligns with the repository's goal of providing insights into the algorithmic implementation. However, for those interested in galaxy morphological classification, we recommend using `pytorch_lightning`, as I do in other ongoing projects.

## Catalogue Information
For information regarding the released catalogue, please visit our Zenodo page (link to be updated).

## Citation

```
@article{XXX/mnras/XXX,
    author = {Ye, Renhao and Shen, Shiyin and de Souza, Rafael S and Xu, Quanfeng and Chen, Mi and Chen, Zhu and Ishida, Emille E O and Krone-Martins, Alberto and Durgesh, Rupesh},
    title = "{From Galaxy Zoo DECaLS to BASS/MzLS: detailed galaxy morphological classification with unsupervised domain adaption}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {XXX},
    number = {X},
    pages = {XXXX-XXXX},
    year = {XXXX},
    month = {XX},
    issn = {XXXX-XXXX},
    doi = {XXXX/mnras/XXXX},
    url = {XXX},
}
```