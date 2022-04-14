# Semantic-aware Triplet Loss for Image Classification
This repository contains the implementation of our paper: Semantic-aware Triplet loss for Image Classification

## Dataset
This repo contains the codes to run our method on CIFAR-10 and CIFAR-100 datasets. 
If you have not downloaded the two datasets, just start running, the datasets will be downloaded automatically.

## Installation
Please install the following libraries 
```
torchmeta 1.5.0
tensorboard 2.4.0
pytorch 1.4.0
```

## Train
To train on CIFAR-100 with Glove as semantic source, just run
```
./scripts/r18_glove_c100.sh
```

## Acknowledgement
Our code is built upon https://github.com/weiaicunzai/pytorch-cifar100 . Thanks for the code!