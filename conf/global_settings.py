""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
CIFAR100_TRAIN_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_MEAN_LY = [0.4914, 0.4822, 0.4465]
IMAGENET_STD_LY = [0.2023, 0.1994, 0.2010]

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160, 200]
# EPOCH = 100
# MILESTONES = [30, 60, 80]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime('%d_%Hh_%Mm_%Ss')

#tensorboard log dir
LOG_DIR = '/temp/guangzhi/exps/sem_reg'





