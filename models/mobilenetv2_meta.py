"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear

class LinearBottleNeck(MetaModule):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = MetaSequential(
            MetaConv2d(in_channels, in_channels * t, 1),
            MetaBatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            MetaConv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            MetaBatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            MetaConv2d(in_channels * t, out_channels, 1),
            MetaBatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, params=None):

        residual = self.residual(x, params=self.get_subdict(params, 'residual'))

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(MetaModule):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = MetaSequential(
            MetaConv2d(3, 32, 1, padding=1),
            MetaBatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = MetaSequential(
            MetaConv2d(320, 1280, 1),
            MetaBatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = MetaConv2d(1280, class_num, 1)

    def forward(self, x, params=None):
        x = self.pre(x, params=self.get_subdict(params, 'pre'))
        x = self.stage1(x, params=self.get_subdict(params, 'stage1'))
        x = self.stage2(x, params=self.get_subdict(params, 'stage2'))
        x = self.stage3(x, params=self.get_subdict(params, 'stage3'))
        x = self.stage4(x, params=self.get_subdict(params, 'stage4'))
        x = self.stage5(x, params=self.get_subdict(params, 'stage5'))
        x = self.stage6(x, params=self.get_subdict(params, 'stage6'))
        x = self.stage7(x, params=self.get_subdict(params, 'stage7'))
        x = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        x = F.adaptive_avg_pool2d(x, 1)
        features = x
        x = self.conv2(x, self.get_subdict(params, 'conv2'))
        x = x.view(x.size(0), -1)

        return features.squeeze(), x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return MetaSequential(*layers)

def mobilenetv2(class_num, pretrained=False):
    model = MobileNetV2(class_num)
    if pretrained:
        state_dict = torch.load("./pretrained_models/mobilenetv2_1.0-0c6065bc.pth")
        model.load_state_dict(state_dict)
    return model