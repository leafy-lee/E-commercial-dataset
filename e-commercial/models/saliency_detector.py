import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import math


def define_salD(input_dim, actType='lrelu', normType='batch', netSalD='denseNet256_36layer'):
    if netSalD == 'CNN_3layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(0, 0, 0), feaChannel=64, reduction=0.5, bottleneck=True,
                          actType=actType, normType=normType, useASPP=True, ASPPsacles=(0, 1, 4, 7))
    elif netSalD == 'denseNet_15layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(2, 2, 2), feaChannel=64, reduction=0.5, bottleneck=True,
                          actType=actType, normType=normType, useASPP=True, ASPPsacles=(0, 1, 4, 7))
    elif netSalD == 'denseNet_20layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(2, 2, 2, 2), feaChannel=64, reduction=0.5,
                          bottleneck=True,
                          actType=actType, normType=normType, useASPP=True, ASPPsacles=(0, 1, 4, 7))
    elif netSalD == 'denseNet_28layer':
        net = SalDetector(input_dim, growthRate=16, EnBlocks=(2, 4, 4, 2), feaChannel=64, reduction=0.5,
                          bottleneck=True,
                          actType=actType, normType=normType, useASPP=True, ASPPsacles=(0, 1, 4, 7))
    else:
        raise NotImplementedError('Saliency detector model name [%s] is not recognized' % netSalD)
    return net


class SalDetector(nn.Module):
    def __init__(self, inputdim=768, growthRate=32, EnBlocks=(2, 2, 2), feaChannel=64, reduction=0.5, bottleneck=True,
                 actType='lrelu', normType='batch', useASPP=True, ASPPsacles=(0, 1, 4, 7)):
        super(SalDetector, self).__init__()
        norm_layer = get_norm_layer(normType)
        act_layer = get_activation_layer(actType)
        nChannels = inputdim
        layers = []
        # layers = [nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)]
        # nChannels = growthRate
        for nLayers in EnBlocks:
            if nLayers == 0:
                CNNlayer = [norm_layer(nChannels),
                            act_layer(),
                            nn.Conv2d(nChannels, nChannels // 2, kernel_size=3, padding=1, bias=False)]
                layers += CNNlayer
                nChannels = nChannels // 2
            else:
                layers += [DenseBlock(nChannels, growthRate, nLayers, reduction, bottleneck, norm_layer, act_layer)]
                nChannels = int(math.floor((nChannels + nLayers * growthRate) * reduction))
        if useASPP:
            layers += [ASPP(nChannels, feaChannel, scales=ASPPsacles)]
        else:
            layers += [nn.Conv2d(nChannels, feaChannel, kernel_size=3, padding=1, bias=False)]
        self.Encoder = nn.Sequential(*layers)
        layers2 = [DecoderBlock(feaChannel, feaChannel // 4, False, norm_layer, act_layer),
                   DecoderBlock(feaChannel // 4, feaChannel // 16, True, norm_layer, act_layer),
                   DecoderBlock(feaChannel // 16, 1, False, norm_layer, act_layer),
                   ]
        self.Decoder = nn.Sequential(*layers2)

    def forward(self, x):
        encoderFea = self.Encoder(x)
        out = self.Decoder(encoderFea)
        # out = torch.squeeze(out)
        out = NormAffine(out, method='one')
        return out


class DecoderBlock(nn.Module):
    def __init__(self, nChannels, nOutChannels, deConv=False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU):
        super(DecoderBlock, self).__init__()
        interChannels = int(math.ceil(nChannels * math.sqrt(nOutChannels / nChannels)))
        if deConv:
            # output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.ConvTranspose2d(nChannels, interChannels, kernel_size=2, stride=2, padding=0, output_padding=0,
                                         bias=False),
                      norm_layer(interChannels),
                      act_layer(),
                      nn.Conv2d(interChannels, nOutChannels, kernel_size=1, bias=False)
                      ]
        else:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.Conv2d(nChannels, interChannels, kernel_size=3, padding=1, bias=False),
                      norm_layer(interChannels),
                      act_layer(),
                      nn.Conv2d(interChannels, nOutChannels, kernel_size=1, bias=False)
                      ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, sn=False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU):
        super(Transition, self).__init__()
        if sn:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.utils.spectral_norm(nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False))
                      # nn.AvgPool2d(2)
                      ]
        else:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
                      # nn.AvgPool2d(2)
                      ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class DenseBlock(nn.Module):
    def __init__(self, nChannels, growthRate, nLayers, reduction, bottleneck=True, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.LeakyReLU, sn=False):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(int(nLayers)):
            if bottleneck:
                layers += [Bottleneck(nChannels, growthRate, sn, norm_layer, act_layer)]
            else:
                layers += [SingleLayer(nChannels, growthRate, sn, norm_layer, act_layer)]
            nChannels += growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        layers += [Transition(nChannels, nOutChannels, sn, norm_layer, act_layer)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, sn=False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        if sn:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.utils.spectral_norm(nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)),
                      norm_layer(interChannels),
                      act_layer(),
                      nn.utils.spectral_norm(nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False))
                      ]
        else:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False),
                      norm_layer(interChannels),
                      act_layer(),
                      nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
                      ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, sn=False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU):
        super(SingleLayer, self).__init__()
        if sn:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.utils.spectral_norm(nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False))
                      ]
        else:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
                      ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = torch.cat((x, out), 1)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256, scales=(0, 1, 4, 7), sn=False):
        super(ASPP, self).__init__()
        self.scales = scales
        for dilate_rate in self.scales:
            if dilate_rate == -1:
                break
            if dilate_rate == 0:
                layers = [nn.AdaptiveAvgPool2d((1, 1))]
                if sn:
                    layers += [nn.utils.spectral_norm(nn.Conv2d(in_channel, depth, 1, 1))]
                else:
                    layers += [nn.Conv2d(in_channel, depth, 1, 1)]
                setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))
            elif dilate_rate == 1:
                if sn:
                    layers = [nn.utils.spectral_norm(nn.Conv2d(in_channel, depth, 1, 1))]
                else:
                    layers = [nn.Conv2d(in_channel, depth, 1, 1)]
                setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))
            else:
                if sn:
                    layers = [nn.utils.spectral_norm(
                        nn.Conv2d(in_channel, depth, 3, 1, dilation=dilate_rate, padding=dilate_rate))]
                else:
                    layers = [nn.Conv2d(in_channel, depth, 3, 1, dilation=dilate_rate, padding=dilate_rate)]
                setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))
        self.conv_1x1_output = nn.Conv2d(depth * len(scales), depth, 1, 1)

    def forward(self, x):
        dilate_outs = []
        for dilate_rate in self.scales:
            if dilate_rate == -1:
                return x
            if dilate_rate == 0:
                layer = getattr(self, 'dilate_layer_{}'.format(dilate_rate))
                size = x.shape[2:]
                tempout = F.interpolate(layer(x), size=size, mode='bilinear', align_corners=True)
                dilate_outs.append(tempout)
            else:
                layer = getattr(self, 'dilate_layer_{}'.format(dilate_rate))
                dilate_outs.append(layer(x))
        out = self.conv_1x1_output(torch.cat(dilate_outs, dim=1))
        return out


####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_activation_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    elif layer_type == 'none':
        nl_layer = None
    else:
        raise NotImplementedError('activitation [%s] is not found' % layer_type)
    return nl_layer


def NormAffine(mat, eps=1e-7,
               method='sum'):  # tensor [batch_size, channels, image_height, image_width] normalize each fea map;
    matdim = len(mat.size())
    if method == 'sum':
        tempsum = torch.sum(mat, dim=(matdim - 1, matdim - 2), keepdim=True) + eps
        out = mat / tempsum
    elif method == 'one':
        (tempmin, _) = torch.min(mat, dim=matdim - 1, keepdim=True)
        (tempmin, _) = torch.min(tempmin, dim=matdim - 2, keepdim=True)
        tempmat = mat - tempmin
        (tempmax, _) = torch.max(tempmat, dim=matdim - 1, keepdim=True)
        (tempmax, _) = torch.max(tempmax, dim=matdim - 2, keepdim=True)
        tempmax = tempmax + eps
        out = tempmat / tempmax
    else:
        raise NotImplementedError('Map method [%s] is not implemented' % method)
    return out
