"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math
from copy import deepcopy
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear


network_param = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicBlock(MetaModule):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        #residual function
        self.residual_function = MetaSequential(
            MetaConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            MetaBatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = MetaSequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = MetaSequential(
                MetaConv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(out_channels * BasicBlock.expansion)
            )
            # print("calling residual function!")

    def forward(self, x, params=None):
        # if len(self.shortcut):
            # print("vanilla shortcut bn:", self.shortcut[1].weight[:5], self.shortcut[1].bias[:5])
        if self.stride != 1 or self.in_channels != BasicBlock.expansion * self.out_channels:
            return nn.ReLU(inplace=True)(self.residual_function(x, self.get_subdict(params, 'residual_function')) + self.shortcut(x, self.get_subdict(params, 'shortcut')))
        else:
            return nn.ReLU(inplace=True)(self.residual_function(x, self.get_subdict(params, 'residual_function')) + self.shortcut(x))

class BottleNeck(MetaModule):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = MetaSequential(
            MetaConv2d(in_channels, out_channels, kernel_size=1, bias=False),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            MetaBatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = MetaSequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = MetaSequential(
                MetaConv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                MetaBatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class mydict(dict):
    
    def __init__(self, d):
        self.data = {}
        self.checked = {}
        for k, v in d.items():
            self.data[k] = v
            self.checked[k] = False

    def __getitem__(self, key):
        self.checked[key] = True
        return self.data[key] 

    def any_unaccessed(self):
        for k, v in self.checked.items():
            if not v:
                return True
        return False

def bn_forward(x, origin, weight, bias):
    if origin.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = origin.momentum

    if origin.training and origin.track_running_stats:
        if origin.num_batches_tracked is not None:
            origin.num_batches_tracked += 1
            if origin.momentum is None:
                exponential_average_factor = 1.0 / float(origin.num_batches_tracked)
            else:
                exponential_average_factor = origin.momentum

    # if origin.training:
        # bn_training = True
    # else:
        # bn_training = (origin.running_mean is None) and (origin.running_var is None)

    return F.batch_norm(x, origin.running_mean, origin.running_var,
        weight, bias, origin.training or not origin.track_running_stats, origin.momentum, origin.eps)

class ResNet(MetaModule):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64

        self.conv1 = MetaSequential(
            MetaConv2d(3, 64, kernel_size=3, padding=1, bias=False),
            MetaBatchNorm2d(64),
            nn.ReLU(inplace=False))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.convs = [None, None, self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MetaLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return MetaSequential(*layers)

    def forward(self, x, params=None, my=False):
        output = self.conv1(x, self.get_subdict(params, "conv1"))
        output = self.conv2_x(output, self.get_subdict(params, "conv2_x"))
        output = self.conv3_x(output, self.get_subdict(params, "conv3_x"))
        output = self.conv4_x(output, self.get_subdict(params, "conv4_x"))
        output = self.conv5_x(output, self.get_subdict(params, "conv5_x"))
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        feat = output
        output = self.fc(output, self.get_subdict(params, "fc"))
        return feat, output



def resnet18(class_num=100, pretrained=False):
    """ return a ResNet 18 object
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], class_num)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet18"])
        # model.load_state_dict(state_dict)
        pretrained_state_list = [(k, v) for k, v in state_dict.items()]
        model_state_list = [(k, v) for k, v in model.named_parameters()]
        
        pretrained_state_list = sorted(pretrained_state_list, key=lambda x:x[1].shape)
        model_state_list = sorted(model_state_list, key=lambda x:x[1].shape)

        # print(len(pretrained_state_list), len(model_state_list))
        for i, k in enumerate(model_state_list):
            print(model_state_list[i][0], pretrained_state_list[i][0])
    
    return model

def resnet34(class_num=100, pretrained=False):
    """ return a ResNet 34 object
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], class_num)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet34"])
        model.load_state_dict(state_dict)
    return model

def resnet50(class_num=100, pretrained=False):
    """ return a ResNet 50 object
    """
    model = ResNet(BottleNeck, [3, 4, 6, 3], class_num)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet50"])
        model.load_state_dict(state_dict)
    return model

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



