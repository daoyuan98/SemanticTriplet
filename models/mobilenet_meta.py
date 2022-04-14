"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear


class DepthSeperabelConv2d(MetaModule):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = MetaSequential(
            MetaConv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            MetaBatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = MetaSequential(
            MetaConv2d(input_channels, output_channels, 1),
            MetaBatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, params=None):
        x = self.depthwise(x, params=self.get_subdict(params, 'depthwise'))
        x = self.pointwise(x, params=self.get_subdict(params, 'pointwise'))

        return x


class BasicConv2d(MetaModule):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = MetaConv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = MetaBatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, params=None):
        x = self.conv(x, params=self.get_subdict(params, 'conv'))
        x = self.bn(x, params=self.get_subdict(params, 'bn'))
        x = self.relu(x)

        return x


class MobileNet(MetaModule):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
       super().__init__()

       alpha = width_multiplier
       self.stem = MetaSequential(
           BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = MetaSequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = MetaSequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = MetaSequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = MetaSequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )
       self.fc = MetaLinear(int(1024 * alpha), class_num)
       self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, params=None):
        x = self.stem(x, params=self.get_subdict(params, "stem"))

        x = self.conv1(x, params=self.get_subdict(params, "conv1"))
        x = self.conv2(x, self.get_subdict(params, "conv2"))
        x = self.conv3(x, self.get_subdict(params, "conv3"))
        x = self.conv4(x, self.get_subdict(params, "conv4"))

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        feat = x
        x = self.fc(x, self.get_subdict(params, "fc"))
        return feat, x


def mobilenet(alpha=1, class_num=100, pretrained=False):
    return MobileNet(alpha, class_num)

