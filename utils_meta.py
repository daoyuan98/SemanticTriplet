""" helper function

author baiyu
"""

import sys
import PIL
import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cifar_dataset


import numpy as np
from copy import deepcopy
class ConsecutiveSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size=128, nclass=100, slice=4):
        self.dataset = dataset
        self.batch_size = batch_size

        # get statistics:
        index_matrix = [[] for i in range(nclass)]
        for i in range(len(dataset)):
            label = dataset[i][1]
            index_matrix[label].append(i)

        self.index_matrix = index_matrix
        self.nclass = nclass
        self.slice = slice

    def __iter__(self):
        index_matrix_ = deepcopy(self.index_matrix)
        index_matrix = []
        for i in range(self.nclass):
            index_matrix.append(np.random.permutation(index_matrix_[i]))

        ret = []
        class_order = np.random.permutation(self.nclass)
        while len(ret) < len(self.dataset):
            for i in range(self.nclass):
                class_to_retrieve = class_order[i]
                sample_left = len(index_matrix[class_to_retrieve])
                if sample_left > 0:
                    ret.extend(index_matrix[class_to_retrieve][:self.slice])
                    index_matrix[class_to_retrieve] = index_matrix[class_to_retrieve][self.slice:]
        
        return iter(ret)
        

    def __len__(self):
        return len(self.dataset)


def get_network(args):
    """ return given network
    """

    
    if args.net == 'resnet18':
        from models.resnet_meta import resnet18
        net = resnet18(class_num=args.class_num, pretrained=args.pretrained)
    elif args.net == 'resnet34':
        from models.resnet_meta import resnet34
        net = resnet34(class_num=args.class_num, pretrained=args.pretrained)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(class_num=args.class_num, pretrained=args.pretrained)
    elif args.net == 'mobilenet':
        from models.mobilenet_meta import mobilenet
        net = mobilenet(class_num=args.class_num, pretrained=args.pretrained)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2_meta import mobilenetv2
        net = mobilenetv2(class_num=args.class_num, pretrained=args.pretrained)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, args=None, batch_size=16, num_workers=2, shuffle=True, dataset="cifar100"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    # if dataset == "cifar100":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    if hasattr(args, "val_size"):
        val_size = args.val_size
    else:
        val_size = 0

    
    if dataset == "cifar100":
        training_dataset = cifar_dataset.CIFAR100(root='./data', split="train", download=True, transform=transform_train, nval=val_size)
    elif dataset == "cifar10":
        training_dataset = cifar_dataset.CIFAR10(root='./data', split="train", download=True, transform=transform_train, nval=val_size)
    # consec_sampler = ConsecutiveSampler(training_dataset)
    training_loader = DataLoader(training_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    # import ipdb; ipdb.set_trace()
    return training_loader


def get_training_dataloader224(mean, std, args=None, batch_size=16, num_workers=2, shuffle=True, dataset="cifar100"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    # if dataset == "cifar100":
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    if hasattr(args, "val_size"):
        val_size = args.val_size
    else:
        val_size = 0

    if dataset == "cifar100":
        training_dataset = cifar_dataset.CIFAR100(root='./data', split="train", download=True, transform=transform_train, nval=val_size)
    elif dataset == "cifar10":
        training_dataset = cifar_dataset.CIFAR10(root='./data', split="train", download=True, transform=transform_train, nval=val_size)

    training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader


def get_valid_dataloader(mean, std, args=None, batch_size=16, num_workers=2, shuffle=True, dataset="cifar100"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if dataset == "cifar100":                      
        val_dataset = cifar_dataset.CIFAR100(root='./data', split="val", download=True, transform=transform_test, nval=args.val_size)
    else:
        val_dataset = cifar_dataset.CIFAR10(root='./data', split="val", download=True, transform=transform_test, nval=args.val_size)

    val_loader = DataLoader(val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return val_loader


def get_test_dataloader(mean, std, args=None, batch_size=16, num_workers=2, shuffle=True, dataset="cifar100"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if dataset == "cifar100":                      
        test_dataset = cifar_dataset.CIFAR100(root='./data', split="test", download=True, transform=transform_test, nval=0)
    else:
        test_dataset = cifar_dataset.CIFAR10(root='./data', split="test", download=True, transform=transform_test, nval=0)

    test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return test_loader


def get_test_dataloader224(mean, std, args=None, batch_size=16, num_workers=2, shuffle=True, dataset="cifar100"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if dataset == "cifar100":                      
        test_dataset = cifar_dataset.CIFAR100(root='./data', split="test", download=True, transform=transform_test, nval=0)
    else:
        test_dataset = cifar_dataset.CIFAR10(root='./data', split="test", download=True, transform=transform_test, nval=0)

    test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
