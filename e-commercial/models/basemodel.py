import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, omit_stride=False,
                 no_res_connect=False, dropout=0., bn_momentum=0.1,
                 batchnorm=None):
        super().__init__()
        self.out_channels = oup
        self.stride = stride
        self.omit_stride = omit_stride
        self.use_res_connect = not no_res_connect and\
            self.stride == 1 and inp == oup
        self.dropout = dropout
        actual_stride = self.stride if not self.omit_stride else 1
        if batchnorm is None:
            def batchnorm(num_features):
                return nn.BatchNorm2d(num_features, momentum=bn_momentum)

        assert actual_stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            modules = [
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, actual_stride, 1,
                          groups=hidden_dim, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.append(nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
        else:
            modules = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, actual_stride, 1,
                          groups=hidden_dim, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.insert(3, nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
            self._initialize_weights()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)