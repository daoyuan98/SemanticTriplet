"""
    LSTM as weighter with 2 loss, each has different hidden state.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmeta.modules import DataParallel

from conf import settings
from utils_meta import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
import logging
import datetime
import shutil
from PIL import Image
import word
from collections import OrderedDict, defaultdict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

global_test_acc = 0.

def soft_relu(x):
    return torch.log(1 + torch.exp(x))

def sim_cosine(x, y):
    return (1 + torch.cosine_similarity(x, y)) / 2.

def sim_euc(x, y):
    d = (x-y).pow(2).sum(dim=-1) + 1e-8
    d = d.sqrt()
    return -d

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        origin_lr = param_group['lr']
        if not isinstance(origin_lr, float):
            origin_lr = origin_lr.item()
        param_group['lr'] = lr


def get_random_val(data_loader):
    n = len(data_loader)
    tar_idx = np.random.randint(0, n-1) 
    for i, (image, label) in enumerate(data_loader):
        if i == tar_idx:
            return image, label


def get_one_label_from_batch(exist_label, label_list):
    while True:
        l = np.random.choice(label_list)
        if not l in exist_label:
            return l


# used for 3 terms
def loss_quadruplet3_r(images, labels, feats, sim_label_lists, similarity_matrix):
    """
        calculating quadruplet loss with 3 terms by randomly selecting samples
    """
    batch_size = images.shape[0]
    class_num = args.class_num

    mask = torch.ones(batch_size).cuda() # to calculate sem loss for this sample or not
    label_list = labels.cpu().numpy().tolist()
    loss = 0.
    label_set = set(label_list)

    summary = defaultdict(list)
    for i, label in enumerate(label_list):
        summary[label].append(i)

    pos_idxs = np.ones(batch_size, dtype=np.int) # dog --> dog
    semi_pos_idxs = np.ones(batch_size, dtype=np.int) # dog --> cat
    neg_idxs = np.ones(batch_size, dtype=np.int) # dog --> fish

    semi_pos_quotient = torch.ones(batch_size).cuda()
    neg_quotient = torch.ones(batch_size).cuda()
    inter_quotient = torch.ones(batch_size).cuda()

    for i, label in enumerate(label_list):
        if len(summary[label]) < 2: # 0 or 1 element, can't calculate loss
            mask[i] = 0.
        else: # can calculate loss only when there is another sample of the same class
            sim_label_list = sim_label_lists[label]
            similarity_list = similarity_matrix[label]

            # select positive sample
            indexes = summary[label]
            while True:
                pos = np.random.choice(indexes)
                if pos != i:
                    pos_idxs[i] = pos
                    break

            l1 = get_one_label_from_batch([label], label_list) 
            l2 = get_one_label_from_batch([label, l1], label_list) # to ensure l1 and l2 are different
            if similarity_list[l1] < similarity_list[l2]:
                semi_pos_label = l2
                neg_label = l1
            else:
                semi_pos_label = l1
                neg_label = l2

            semi_pos_idxs[i] = label_list.index(semi_pos_label)
            neg_idxs[i] = label_list.index(neg_label)

            semi_pos_quotient[i] = similarity_list[label] / similarity_list[semi_pos_label]
            neg_quotient[i] = similarity_list[label] / similarity_list[neg_label]
            inter_quotient[i] = similarity_list[semi_pos_label] / similarity_list[neg_label]
            
            if semi_pos_quotient[i] == neg_quotient[i]: # when there is no difference, do not calculate the loss
                mask[i] = 0.

    s_app = sim_cosine(feats, feats[pos_idxs]) # similarity between anchor (a) and positive positive (pp)
    s_apn = sim_cosine(feats, feats[semi_pos_idxs]) # similarity between anchor (a) and positive negative (pn)
    s_an  = sim_cosine(feats, feats[neg_idxs]) # similarity between anchor (a) and negative (n)

    diff_pos_semipos = s_app - s_apn
    diff_pos_neg = s_app - s_an
    diff_semipos_neg = s_apn - s_an

    loss1 = F.relu(torch.tanh(args.beta * torch.log(semi_pos_quotient)) - diff_pos_semipos) # term 2
    loss2 = F.relu(torch.tanh(args.beta * torch.log(neg_quotient)) - diff_pos_neg)  # term 1
    loss3 = F.relu(torch.tanh(args.beta * torch.log(inter_quotient)) - diff_semipos_neg)    # term 3

    losses = [loss1, loss2, loss3]
    for i in range(len(losses)):
        losses[i] = losses[i] * mask
        losses[i] = losses[i].sum() / (mask.sum().item())
    return losses, mask.sum()



def train(epoch, args, similarity_matrix=None, sim_label_lists=None):
    torch.autograd.set_detect_anomaly(True)
    start = time.time()
    weighter.eval()
    net.train()
    nbatch = len(cifar_training_loader) - 1
    update_iters = [int(i * nbatch / args.nupdate) for i in range(1, args.nupdate)] + [nbatch]
    # print(update_iters)
    for batch_index, (images, labels) in enumerate(cifar_training_loader):
        label_list = labels.numpy().tolist()
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        feats, outputs = net(images)
        class_loss = loss_function(outputs, labels)
        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1

        (sl1, sl2, sl3), nvalid = loss_quadruplet3_r(images, labels, feats, sim_label_lists, similarity_matrix)

        if batch_index in update_iters:
            weighter.train()
            weighter.zero_grad()
            
            input_loss = torch.cat([class_loss[None, None], sl1[None, None], sl2[None, None], sl3[None, None]], dim=1)
            gammas = [args.weighter_gamma1, args.weighter_gamma2, args.weighter_gamma3]
            w1, w2, w3 = weighter(input_loss, gammas)
            # print(sl1, sl2, sl3)
            grads = torch.autograd.grad(class_loss + w1 * sl1 + w2 * sl2 + w3 * sl3, net.parameters(), retain_graph=True, create_graph=True)

            fast_weights = OrderedDict((name, param - optimizer.param_groups[0]['lr'] * grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
            val_data, val_label = get_random_val(cifar_training_loader)
            if args.gpu:
                val_data = val_data.cuda()
                val_label = val_label.cuda()
            feats, predict = net(val_data, fast_weights)
            meta_loss = loss_function(predict, val_label)
            logger.info("meta loss: {:.4f}".format(meta_loss.item()))

            weighter_grads = torch.autograd.grad(meta_loss, weighter.parameters(), retain_graph=True, allow_unused=True)
            for p, g in zip(weighter.parameters(), weighter_grads): 
                p.grad = g.detach()

            weighter_optimizer.step()
            weighter.eval()
            
            input_loss = torch.cat([class_loss[None, None], sl1[None, None], sl2[None, None], sl3[None, None]], dim=1).detach()
            w1, w2, w3 = weighter(input_loss, [args.weighter_gamma1, args.weighter_gamma2, args.weighter_gamma3], detach=True)
            writer.add_scalar('Train/predicted_weight1', w1.item(), n_iter)
            writer.add_scalar('Train/predicted_weight2', w2.item(), n_iter)
            writer.add_scalar('Train/predicted_weight3', w3.item(), n_iter)
        else:
            input_loss = torch.cat([class_loss[None, None], sl1[None, None], sl2[None, None], sl3[None, None]], dim=1).detach()
            w1, w2, w3 = weighter(input_loss, [args.weighter_gamma1, args.weighter_gamma2, args.weighter_gamma3])
            w1 = w1.detach()
            w2 = w2.detach()
            w3 = w3.detach()
        
        loss = class_loss + w1 * sl1 + w2 * sl2 + w3 * sl3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % args.print_every == 0:
            logger.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}] CLs:{:0.4f} Sl1:{:0.4f} Sl2:{:0.4f} Sl3:{:0.4f} SW1:{:0.4f} SW2:{:0.4f} SW3:{:0.4f} LR:{:0.4f}'.format(
                class_loss.item(),
                sl1.item(),
                sl2.item(),
                sl3.item(),
                w1[0].item(),
                w2[0].item(),
                w3[0].item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar_training_loader.dataset)
            ))

        writer.add_scalar('Train/class_loss', loss.item(), n_iter)
        writer.add_scalar('Train/sem_loss1', sl1.item(), n_iter)
        writer.add_scalar('Train/sem_loss2', sl2.item(), n_iter)
        writer.add_scalar('Train/sem_loss3', sl3.item(), n_iter)
                
    finish = time.time()
    logger.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch):
    global global_test_acc
    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        feats, outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    logger.info('Evaluating Network.....')
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar_test_loader.dataset),
        correct.float() / len(cifar_test_loader.dataset),
        finish - start
    ))
    logger.info('\n')

    #add informations to tensorboard
    test_acc = correct.float() / len(cifar_test_loader.dataset)
    global_test_acc = max(global_test_acc, test_acc)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar_test_loader.dataset), epoch)

    return correct.float() / len(cifar_test_loader.dataset)


def initialize_with_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gen_selection_pool(mat):
    assert mat.shape[0] == mat.shape[1]
    ret = np.zeros_like(mat, dtype=np.int)
    n_class = mat.shape[0]
    for i in range(n_class):
        l = sorted(range(n_class), key=lambda k:mat[i][k])
        ret[i] = l
    return ret


class LSTMCell(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.fc_i2h = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 4 * hidden_size)
        )
        self.fc_h2h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size * 4)
        )

    def forward(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)

        x = i2h + h2h
        
        gates = x.split(self.hidden_size, 1)
        in_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1])
        out_gate = torch.sigmoid(gates[2])
        in_transform = torch.tanh(gates[3])
        cx = forget_gate * cx + in_gate * in_transform

        hx = out_gate * torch.tanh(cx)
        return hx, cx


class Weighter(nn.Module):
    def __init__(self, hidden_size=40, num_tloss=3): 
        super(Weighter, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = LSTMCell(num_tloss+1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_tloss)
        self.num_tloss = num_tloss
        self.hx = torch.zeros(1, hidden_size).cuda()
        self.cx = torch.zeros(1, hidden_size).cuda()

    def forward(self, x, gamma, detach=False):
        hx, cx = self.layer1(x, (self.hx, self.cx))
        x_ = self.layer2(hx)
        out = torch.sigmoid(x_)
        if detach:
            self.hx = hx.detach()
            self.cx = cx.detach()
        else:
            self.hx = hx
            self.cx = cx
        return out[:, 0] * gamma[0], out[:, 1] * gamma[1], out[:, 2] * gamma[2]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # on basic experimental setting
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-seed', type=int, default=19, help='random seed')
    parser.add_argument('-dataset', type=str, default='cifar100', help='choose which dataset to train on')
    parser.add_argument('-print_every', type=int, default=50)

    # on semantic regularizer
    parser.add_argument('-similarity_source', type=str, default="wordnet")
    parser.add_argument('-beta', type=float, default=0.1, help="beta value to smooth tanh function")
    parser.add_argument('-weighter_gamma1', type=float, default=2.0, help="gamma value of the loss weight")
    parser.add_argument('-weighter_gamma2', type=float, default=2.0, help="2nd gamma value of the loss weight")
    parser.add_argument('-weighter_gamma3', type=float, default=2.0, help="3rd gamma value of the loss weight")
    parser.add_argument('-nupdate', type=int, default=2, help="number of updates of meta-learner")

    # on running setting
    parser.add_argument('-exp_name', type=str, default='default_exp', help='default experiments saving name')
    parser.add_argument('-best', type=bool, default=False, help='whether use best search hyparam')
    parser.add_argument('-pretrained', type=bool, default=False)

    args = parser.parse_args()
    args.class_num = 100 if args.dataset == "cifar100" else 10

    initialize_with_seed(args.seed)

    net = get_network(args)
    weighter = Weighter(hidden_size=40, num_tloss=3)

    if args.gpu:
        weighter = weighter.cuda()

    cifar_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        args=args,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset=args.dataset
    )

    cifar_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        args=args,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset=args.dataset
    )
    
    sim_mat = word.get_similarity_matrix(args.dataset, args.similarity_source)
    sim_list = gen_selection_pool(sim_mat)

    if args.net == 'mobilenet' or args.net == 'mobilenetv2':
        args.lr = args.lr / 2
        settings.EPOCH = 300
        args.stop_epoch = 150

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    weighter_optimizer = optim.Adam(weighter.parameters(), lr=1e-3, weight_decay=1e-4)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar_training_loader)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    save_dir = os.path.join(settings.LOG_DIR, args.exp_name, settings.TIME_NOW)

    best_acc = 0.0    
    acc = eval_training(0)
    for epoch in range(1, settings.EPOCH+1):
        train(epoch, args, sim_mat, sim_list)
        acc = eval_training(epoch)
        
        # add save file
        file_name = "last.pth.tar"
        checkpoint = {
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "weighter": weighter.state_dict(),
            "acc": acc,
            "optimizer": optimizer.state_dict()
        }
        file_path = os.path.join(save_dir, file_name)
        torch.save(checkpoint, file_path)
        if acc > best_acc:
            best_acc = acc
            shutil.copy(file_path, os.path.join(save_dir, 'best.pth.tar'))

        train_scheduler.step()
            
    logger.info(args)
    logger.info("Best test acc: {}".format(global_test_acc))
