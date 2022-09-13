import torch
import torch.nn as nn
import functools
import random
import torch.nn.functional as F
import math
from torchvision import models
import pdb

####################################################################
#------------------------- Saliency Detector --------------------------
####################################################################

def define_salD(input_dim, actType='lrelu', normType='batch', netSalD='denseNet256_36layer',debug=False):
    if netSalD == 'denseNet256_36layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(2, 3, 8), feaChannel=64, reduction=0.5, bottleneck=True,
                          actType=actType, normType=normType, useASPP=True, ASPPsacles=(0, 1, 4, 8, 12),lowres=False,debug = debug)
    elif netSalD == 'denseNet256_60layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(3, 6, 16), feaChannel=128, reduction=0.5,
                          bottleneck=True, actType=actType, normType=normType, useASPP=True,
                          ASPPsacles=(0, 1, 4, 8, 12),lowres=False,debug = debug)
    elif netSalD == 'denseNet128_36layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(5, 8), feaChannel=64, reduction=0.5,
                          bottleneck=True, actType=actType, normType=normType, useASPP=True,
                          ASPPsacles=(0, 1, 4, 8),lowres=True,debug = debug)
    elif netSalD == 'denseNet128_60layer':
        net = SalDetector(input_dim, growthRate=32, EnBlocks=(9, 16), feaChannel=128, reduction=0.5,
                          bottleneck=True, actType=actType, normType=normType, useASPP=True,
                          ASPPsacles=(0, 1, 4, 8),lowres=True,debug = debug)
    else:
        raise NotImplementedError('Saliency detector model name [%s] is not recognized' % netSalD)
    return net


class SalDetector(nn.Module):
    def __init__(self, inputdim, growthRate = 32, EnBlocks = (2,3,8), feaChannel=64, reduction=0.5, bottleneck=True,
                 actType='lrelu', normType='batch', useASPP=True, ASPPsacles=(0, 1, 4, 8, 12), lowres=False,debug = False):
        super(SalDetector, self).__init__()
        self.debug = debug
        norm_layer = get_norm_layer(normType)
        act_layer = get_activation_layer(actType)
        nChannels = inputdim
        layers = [nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)]
        nChannels = growthRate
        for nLayers in EnBlocks:
          layers += [DenseBlock(nChannels, growthRate, nLayers, reduction, bottleneck, norm_layer, act_layer)]
          nChannels = int(math.floor((nChannels + nLayers*growthRate)*reduction))
        if useASPP:
            layers += [ASPP(nChannels, feaChannel, scales=ASPPsacles)]
        else:
            layers += [nn.Conv2d(nChannels, feaChannel, kernel_size=3, padding=1, bias=False)]
        self.Encoder = nn.Sequential(*layers)
        layers2 = [DecoderBlock(feaChannel, feaChannel//4, False, norm_layer, act_layer),
                   DecoderBlock(feaChannel//4, feaChannel//16, not lowres, norm_layer, act_layer),
                   DecoderBlock(feaChannel//16, 1, False, norm_layer, act_layer),
                   ]
        self.Decoder = nn.Sequential(*layers2)

    def forward(self, x):
        encoderFea = self.Encoder(x)
        out = self.Decoder(encoderFea)
        #out = torch.squeeze(out)
        out = NormAffine(out, method='one')
        if self.debug:
            nansum = torch.sum(torch.isinf(x))
            if nansum > 0:
                print('the input of saliency detector havs %d nan' % nansum)
            assert nansum == 0
            nansum = torch.sum(torch.isinf(out))
            if nansum > 0:
                print('the output of saliency detector havs %d nan' % nansum)
            assert nansum==0

        return out

####################################################################
#------------------------- Encoder --------------------------
####################################################################
def define_E(inputdim=6, zdim=4, nef=64, actType='lrelu', normType='instance', vaeLike=True, sn=False, netE = 'denseE256',debug=False):
    if netE == 'nlayersE128':
        net = E_NLayers(inputdim, output_nc=zdim, nef=nef, n_layers=4,
                 actType=actType, normType=normType, vaeLike=vaeLike, sn=sn, debug=debug)
    elif netE == 'nlayersE256':
        net = E_NLayers(inputdim, output_nc=zdim, nef=nef, n_layers=5,
                        actType=actType, normType=normType, vaeLike=vaeLike, sn=sn, debug=debug)
    elif netE == 'denseE128':
        net = E_denseNets(inputdim, output_nc=zdim, nef=nef, growthRate=64, EnBlocks=(3,3,3),
                 reduction=0.5, bottleneck=True, actType=actType, normType=normType, vaeLike=vaeLike, sn=sn, debug=debug)
    elif netE == 'denseE256':
        net = E_denseNets(inputdim, output_nc=zdim, nef=nef, growthRate=64, EnBlocks=(3,3,3,3),
                 reduction=0.5, bottleneck=True, actType=actType, normType=normType, vaeLike=vaeLike, sn=sn, debug=debug)
    elif netE == 'resnetE128':
        net = E_ResNet(inputdim, zdim, nef, n_blocks=4, actType=actType, normType=normType, vaeLike=vaeLike)
    elif netE == 'resnetE256':
        net = E_ResNet(inputdim, zdim, nef, n_blocks=5, actType=actType, normType=normType, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % netE)
    return net

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4, normType='batch',
                 actType='lrelu', vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        norm_layer = get_norm_layer(normType)
        nl_layer = get_activation_layer(actType)
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=4, nef=64, n_layers=5,     #5 5:256 4:128
                 actType='lrelu', normType='batch', vaeLike=True, sn=False,  debug=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike
        self.input_nc = input_nc
        self.debug = debug
        norm_layer = get_norm_layer(normType)
        act_layer = get_activation_layer(actType)
        if sn:
            sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, nef, kernel_size=4,
                                  stride=2, padding=1)), act_layer()]
        else:
            sequence = [nn.Conv2d(input_nc, nef, kernel_size=4,
                                  stride=2, padding=1), act_layer()]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            tempconv = nn.Conv2d(nef * nf_mult_prev, nef * nf_mult,
                          kernel_size=4, stride=2, padding=1)
            if sn:
                tempconv = nn.utils.spectral_norm(tempconv)
            sequence += [tempconv]
            if norm_layer is not None:
                sequence += [norm_layer(nef * nf_mult)]
            sequence += [act_layer()]
        sequence += [nn.AdaptiveAvgPool2d(1)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(nef * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(nef * nf_mult, output_nc)])
    def forward(self, x):
        assert x.size()[1] == self.input_nc
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size()[0], -1)
        output = self.fc(conv_flat)
        if self.debug:
            nansum = torch.sum(torch.isinf(x))
            nansum2 = torch.sum(torch.isnan(x))
            if nansum2 > 0 or nansum > 0:
                print('the input of encoder has %d nan and %d inf' % (nansum2, nansum))
            assert nansum == 0 and nansum2==0
            nansum = torch.sum(torch.isinf(output))
            nansum2= torch.sum(torch.isnan(output))
            if nansum2 > 0 or nansum > 0:
                print('the output of encoder has %d nan and %d inf' % (nansum2, nansum))
            assert nansum == 0 and nansum2 == 0
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

class E_denseNets(nn.Module):
    def __init__(self, input_nc, output_nc=4, nef=64, growthRate = 64, EnBlocks = (3,3,3,3),
                 reduction=0.5, bottleneck=True,  actType='lrelu', normType='batch', vaeLike=True, sn=False, debug=False):
        super(E_denseNets, self).__init__()
        self.vaeLike = vaeLike
        self.input_nc = input_nc
        self.debug = debug
        norm_layer = get_norm_layer(normType)
        act_layer = get_activation_layer(actType)
        nChannels = input_nc
        if sn:
            layers = [nn.utils.spectral_norm(nn.Conv2d(nChannels, nef, kernel_size=4, stride=2, padding=1))]
        else:
            layers = [nn.Conv2d(nChannels, nef, kernel_size=4, stride=2, padding=1)]

        nChannels = nef
        for it, nLayers in enumerate(EnBlocks):
            layers += [DenseBlock(nChannels, growthRate, nLayers, reduction, bottleneck, norm_layer, act_layer, sn, debug, it)]
            nChannels = int(math.floor((nChannels + nLayers * growthRate) * reduction))
            layers += [nn.AdaptiveAvgPool2d(1)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(*[nn.Linear(nChannels, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(nChannels, output_nc)])
    def forward(self, x):
        assert x.size()[1] == self.input_nc
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size()[0], -1)
        output = self.fc(conv_flat)
        if self.debug:
            nansum = torch.sum(torch.isinf(x))
            nansum2 = torch.sum(torch.isnan(x))
            if nansum > 0 or nansum2 > 0 :
                print('the input of encoder has %d nan and %d inf' % (nansum2,nansum))
            assert nansum == 0 and nansum2 == 0
            nansum = torch.sum(torch.isinf(output))
            nansum2 = torch.sum(torch.isnan(output))
            if nansum > 0 or nansum2 > 0:
                print('the output of encoder has %d nan and %d inf' % (nansum2,nansum))
            assert nansum == 0 and nansum2 == 0
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


####################################################################
#------------------------- Discriminators --------------------------
####################################################################
def define_D(input_dim=3, ch=32, actType='lrelu', normType='instance', sn=True, dilate_scales=(1,2,4,6), netD = 'multiDis256', salshare=False, cond=False, debug=False):
    if salshare:
        if netD == 'multiDis128':
            net = MultiScaleDisWithSalDetector(input_dim=input_dim, ch=ch, n_scale=3, n_layer=3,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        elif netD == 'multiDis256':
            net = MultiScaleDisWithSalDetector(input_dim=input_dim, ch=ch, n_scale=3, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        elif netD == 'singleDis128':
            net = MultiScaleDisWithSalDetector(input_dim=input_dim, ch=ch, n_scale=1, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        elif netD == 'singleDis256':
            net = MultiScaleDisWithSalDetector(input_dim=input_dim, ch=ch, n_scale=1, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond,debug=debug)
        elif netD == 'OneMap256':
            net = MultiScaleDisWithSalDetector2(input_dim=input_dim, ch=ch, n_scale=1, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    else:
        if netD == 'multiDis128':
            net = MultiScaleDis(input_dim=input_dim, ch=ch, n_scale=3, n_layer=3,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        elif netD == 'multiDis256':
            net = MultiScaleDis(input_dim=input_dim, ch=ch, n_scale=3, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        elif netD == 'singleDis128':
            net = MultiScaleDis(input_dim=input_dim, ch=ch, n_scale=1, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        elif netD == 'singleDis256':
            net = MultiScaleDis(input_dim=input_dim, ch=ch, n_scale=1, n_layer=4,
                   actType=actType, normType=normType, sn=sn, dilate_scales=dilate_scales, cond=cond, debug=debug)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return net


class MultiScaleDis(nn.Module):
  def __init__(self, input_dim=3, ch=32, n_scale=3, n_layer=3,
               actType='lrelu', normType='instance', sn=True, dilate_scales=(1,2,4,6), cond=False, debug=False):
    super(MultiScaleDis, self).__init__()
    self.n_scale = n_scale
    self.debug = debug
    norm_layer = get_norm_layer(normType)
    act_layer = get_activation_layer(actType)
    self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    self.cond = cond
    if cond:
        input_dim = input_dim + 1
    for i in range(self.n_scale):
        disc = self.singlescale_net(input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales)
        setattr(self, 'disc_{}'.format(i), disc)

  def singlescale_net(self, input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales):
    model = []
    if sn:
        model += [nn.utils.spectral_norm(nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1))]
    else:
        model += [nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)]
    prech = ch
    tch = min(prech * 2, ch*8)
    for _ in range(1, n_layer):
        model += [NormActConv(prech, tch, 3, 2, 1, act_layer, norm_layer, sn)]
        prech = tch
        tch = min(prech * 2, ch*8)
    model += [ASPP(prech, prech // 2, dilate_scales, sn)]
    model += [NormActConv(prech // 2, 1, 3, 1, 1, act_layer, norm_layer, sn)]
    return nn.Sequential(*model)

  def forward(self, x):
    outs = []
    x2 = x
    nanoutD = 0
    infoutD = 0
    for i in range(self.n_scale):
        disc = getattr(self, 'disc_{}'.format(i))
        outD = disc(x)
        outs.append(outD)
        x = self.downsample(x)
        if self.debug:
            infsum = torch.sum(torch.isinf(outD))
            nansum = torch.sum(torch.isnan(outD))
            nanoutD = nanoutD + nansum
            infoutD = infoutD + infsum

    if self.debug:
        nansum = torch.sum(torch.isnan(x2))
        infsum = torch.sum(torch.isinf(x2))
        assert nansum == 0 and infsum == 0
        if nansum > 0 or infsum > 0:
            print('the input of discriminator has %d nan and %d inf' % (nansum,infsum))
        if nanoutD > 0 or infoutD > 0:
            print('the output of discriminator has %d nan and %d inf' % (nanoutD, infoutD))
        assert nanoutD == 0 and infoutD == 0
    return outs

class MultiScaleDisWithSalDetector2(nn.Module):
  def __init__(self, input_dim=3, ch=32, n_scale=3, n_layer=3,
               actType='lrelu', normType='instance', sn=True, dilate_scales=(1,2,4,6),cond=False,debug=False):
    super(MultiScaleDisWithSalDetector2, self).__init__()
    self.n_scale = n_scale
    self.debug = debug
    norm_layer = get_norm_layer(normType)
    act_layer = get_activation_layer(actType)
    self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    self.cond = cond
    sumch = 0
    if cond:
        input_dim = input_dim + 1
    for i in range(self.n_scale):
        en, disD, prech = self.singlescale_net(input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales)
        setattr(self, 'en_{}'.format(i), en)
        setattr(self, 'disD_{}'.format(i), disD)
        sumch = sumch + prech
    sallayers = [NormActConv(sumch, sumch//2, 3, 1, 1, act_layer, norm_layer, sn),
               NormActConv(sumch//2, 1, 3, 1, 1, act_layer, norm_layer, sn)]
    self.saldector = nn.Sequential(*sallayers)

      # for map in salOuts:
      #     tempmap = F.interpolate(map, size=(gtmap.size()[2], gtmap.size()[3]), mode='bilinear')
      #     kl_loss += networks.KL_loss(tempmap, gtmap)
      #     salmaps += tempmap

  def singlescale_net(self, input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales):
    model = []
    if sn:
        model += [nn.utils.spectral_norm(nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1))]
    else:
        model += [nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)]
    prech = ch
    tch = min(prech * 2, ch*8)
    for _ in range(1, n_layer):
        model += [NormActConv(prech, tch, 3, 2, 1, act_layer, norm_layer, sn)]
        prech = tch
        tch = min(prech * 2, ch*8)
    model += [ASPP(prech, prech // 2, dilate_scales, sn)]
    singleEncoder = nn.Sequential(*model)
    prech = prech // 2
    disDecoder = NormActConv(prech, 1, 3, 1, 1, act_layer, norm_layer, sn)
    # salDecoder = NormActConv(prech // 2, 1, 3, 1, 1, act_layer, norm_layer, sn)
    return singleEncoder, disDecoder, prech

  def forward(self, x):
    disOuts = []
    salfeas = []
    x2 = x
    nanoutD = 0
    nanoutSal = 0
    infoutD = 0
    infoutSal = 0
    for i in range(self.n_scale):
        en = getattr(self, 'en_{}'.format(i))
        disD = getattr(self, 'disD_{}'.format(i))
        feas = en(x)
        disoutD = disD(feas)
        #saloutD = NormAffine(salD(feas), method='one')
        tempfea = F.interpolate(feas, size=(x2.size()[2]//4, x2.size()[3]//4), mode='bilinear')
        disOuts.append(disoutD)
        salfeas.append(tempfea)
        x = self.downsample(x)
        if self.debug:
            nansum = torch.sum(torch.isnan(disoutD))
            nanoutD = nanoutD + nansum
            infsum = torch.sum(torch.isinf(disoutD))
            infoutD = infoutD + infsum
    catfea = torch.cat(salfeas, 1)
    salmap = NormAffine(self.saldector(catfea), method='one')
    if self.debug:
        nansum = torch.sum(torch.isnan(x2))
        infsum = torch.sum(torch.isinf(x2))
        nanoutSal = torch.sum(torch.isnan(salmap))
        infoutSal = torch.sum(torch.isinf(salmap))
        if nansum > 0 or infsum > 0:
            print('the input of discriminator has %d nan and %d inf' % (nansum, infsum))
        if nanoutD > 0 or infoutD > 0:
            print('the output of discriminator has %d nan and %d inf' % (nanoutD, infoutD))
        if nanoutSal > 0 or infoutSal > 0:
            print('the salmap of discriminator has %d nan and %d inf' % (nanoutSal, infoutSal))
    return disOuts, salmap

class MultiScaleDisWithSalDetector(nn.Module):
      def __init__(self, input_dim=3, ch=32, n_scale=3, n_layer=3,
                   actType='lrelu', normType='instance', sn=True, dilate_scales=(1, 2, 4, 6), cond=False, debug=False):
          super(MultiScaleDisWithSalDetector, self).__init__()
          self.n_scale = n_scale
          self.debug = debug
          norm_layer = get_norm_layer(normType)
          act_layer = get_activation_layer(actType)
          self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
          self.cond = cond
          if cond:
              input_dim = input_dim + 1
          for i in range(self.n_scale):
              en, disD, salD = self.singlescale_net(input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales)
              setattr(self, 'en_{}'.format(i), en)
              setattr(self, 'disD_{}'.format(i), disD)
              setattr(self, 'salD_{}'.format(i), salD)

      def singlescale_net(self, input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales):
          model = []
          if sn:
              model += [nn.utils.spectral_norm(nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1))]
          else:
              model += [nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)]
          prech = ch
          tch = min(prech * 2, ch * 8)
          for _ in range(1, n_layer):
              model += [NormActConv(prech, tch, 3, 2, 1, act_layer, norm_layer, sn)]
              prech = tch
              tch = min(prech * 2, ch * 8)
          model += [ASPP(prech, prech // 2, dilate_scales, sn)]
          singleEncoder = nn.Sequential(*model)
          disDecoder = NormActConv(prech // 2, 1, 3, 1, 1, act_layer, norm_layer, sn)
          salDecoder = NormActConv(prech // 2, 1, 3, 1, 1, act_layer, norm_layer, sn)
          return singleEncoder, disDecoder, salDecoder

      def forward(self, x):
          disOuts = []
          salOuts = []
          x2 = x
          nanoutD = 0
          nanoutSal = 0
          infoutD = 0
          infoutSal = 0
          for i in range(self.n_scale):
              en = getattr(self, 'en_{}'.format(i))
              disD = getattr(self, 'disD_{}'.format(i))
              salD = getattr(self, 'salD_{}'.format(i))
              feas = en(x)
              disoutD = disD(feas)
              saloutD = NormAffine(salD(feas), method='one')
              disOuts.append(disoutD)
              salOuts.append(saloutD)
              x = self.downsample(x)
              if self.debug:
                  nansum = torch.sum(torch.isnan(disoutD))
                  nanoutD = nanoutD + nansum
                  nansum = torch.sum(torch.isnan(saloutD))
                  nanoutSal = nanoutSal + nansum
                  infsum = torch.sum(torch.isinf(disoutD))
                  infoutD = infoutD + infsum
                  infsum = torch.sum(torch.isinf(saloutD))
                  infoutSal = infoutSal + infsum

          if self.debug:
              nansum = torch.sum(torch.isnan(x2))
              infsum = torch.sum(torch.isinf(x2))
              if nansum > 0 or infsum > 0:
                  print('the input of discriminator has %d nan and %d inf' % (nansum, infsum))
              if nanoutD > 0 or infoutD > 0:
                  print('the output of discriminator has %d nan and %d inf' % (nanoutD, infoutD))
              if nanoutSal > 0 or infoutSal > 0:
                  print('the salmap of discriminator has %d nan and %d inf' % (nanoutSal, infoutSal))
          return disOuts, salOuts
####################################################################
#------------------------- Generator --------------------------
####################################################################

def define_G(input_nc, output_nc, nc, nz, ngf, normType, netG='unet128', sn = False, debug=False):
        if netG == 'unet128':
            net = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=7, addcmap=False, addz=False, nc=nc, nz=nz, ngf=ngf,
                      normType=normType, use_dropout=False, sn=sn, debug=debug)
        elif netG == 'unet256':
            net = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=8, addcmap=False, addz=False, nc=nc, nz=nz, ngf=ngf,
                      normType=normType, use_dropout=False, sn=sn, debug=debug)
        elif netG == 'unet128_withcmap':
            net = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=7, addcmap=True, addz=False, nc=nc, nz=nz, ngf=ngf,
                      normType=normType, use_dropout=False, sn=sn,debug=debug)
        elif netG == 'unet256_withcmap':
            net = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=8, addcmap=True, addz=False, nc=nc, nz=nz, ngf=ngf,
                      normType=normType, use_dropout=False,sn=sn, debug=debug)
        elif netG == 'unet128_withcmapZ':
            net = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=7, addcmap=True, addz=True, nc=nc, nz=nz, ngf=ngf,
                      normType=normType, use_dropout=False, sn=sn,debug=debug)
        elif netG == 'unet256_withcmapZ':
            net = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=8, addcmap=True, addz=True, nc=nc, nz=nz, ngf=ngf,
                      normType=normType, use_dropout=False, sn=sn,debug=debug)
        elif netG == 'resnet128':
            net = ResnetGenerator(input_nc=input_nc, output_nc=output_nc,ifdomw=True, n_blocks=6, addcmap=True, addz=False, nc=nc, nz=nz, ngf=ngf,
                      normType=normType,debug=debug)
        elif netG == 'resnet256':
            net = ResnetGenerator(input_nc=input_nc, output_nc=output_nc, ifdomw=True,n_blocks=7, addcmap=True, addz=False, nc=nc, nz=nz, ngf=ngf,
                      normType=normType,debug=debug)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

        return net


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ifdomw=True, sn=False, addcmap=False, addz=False, nc=1, nz=4, ngf=64, normType='instance', n_blocks=6, padding_type='reflect', debug=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        norm_layer = get_norm_layer(normType)
        self.nc = nc
        self.nz = nz
        self.debug = debug
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        input_nc = input_nc + nc + nz
        if sn:
            model = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias))]
        else:
            model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias)]
        model += [
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if ifdomw:
                downconv = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
            else:
                downconv = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if sn:
                downconv = nn.utils.spectral_norm(downconv)
            model += [downconv,
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, sn=sn, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if ifdomw:
                upconv =nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
            else:
                upconv = nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias)
            if sn:
                upconv = nn.utils.spectral_norm(upconv)

            model += [upconv,
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, bias=use_bias))]
        else:
            model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, bias=use_bias)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x, cmap=None, z=None):
        if self.nc >0:
            assert cmap.size()[1] == self.nc and x.size()[2] == cmap.size()[2] and x.size()[3] == cmap.size()[3]
        if self.nz > 0:
            assert z.size()[1] == self.nz
        z_img = z.view(z.size()[0], z.size()[1], 1, 1).expand(z.size()[0], z.size()[1], x.size()[2],
                                                              x.size()[3])
        x = torch.cat([x,cmap, z_img], 1)
        out = self.model(x)

        if self.debug:
            nansum = torch.sum(torch.isnan(out))
            infsum = torch.sum(torch.isinf(out))
            if nansum > 0 or infsum > 0:
                print('the input of discriminator has %d nan and %d inf' % (nansum, infsum))
            assert nansum == 0 and infsum == 0
        return out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, sn, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, sn, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, sn, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if sn:
            conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        conv_block += [norm_layer(dim), nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if sn:
            conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out



class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, addcmap=False, addz=False, nc=1, nz=4, ngf=64, normType='instance',  use_dropout=False, sn=False, debug=False):
        """Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            nc (int)  -- the number of channels of the conditional map
            nz (int) -- the number of channels of the noise vector
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.nc = nc
        self.nz = nz
        norm_layer = get_norm_layer(normType)
        # the outer_nc (==input_nc) should equal to the inner_nc of last block (encoder stage)
        unet_block = UnetBlockWithCmapAndNoise(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, addcmap=addcmap, addz=addz, nc=nc, nz=nz, sn=sn, debug=debug)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetBlockWithCmapAndNoise(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, addcmap=addcmap, addz=addz, nc=nc, nz=nz,sn=sn, debug=debug)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetBlockWithCmapAndNoise(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, addcmap=addcmap, addz=addz, nc=nc, nz=nz,sn=sn, debug=debug)
        unet_block = UnetBlockWithCmapAndNoise(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, addcmap=addcmap, addz=addz, nc=nc, nz=nz,sn=sn, debug=debug)
        unet_block = UnetBlockWithCmapAndNoise(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, addcmap=addcmap, addz=addz, nc=nc, nz=nz, sn=sn,debug=debug)
        self.model = UnetBlockWithCmapAndNoise(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, addcmap=addcmap, addz=addz, nc=nc, nz=nz, sn=sn,debug=debug)  # add the outermost layer

    def forward(self, x, cmap=None, z=None):
        if self.nc >0:
            assert cmap.size()[1] == self.nc and x.size()[2] == cmap.size()[2] and x.size()[3] == cmap.size()[3]
        if self.nz > 0:
            assert z.size()[1] == self.nz
        return self.model(x,cmap,z)


class UnetBlockWithCmapAndNoise(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 addcmap = False, addz = False, nc = 1, nz = 4, sn=False, debug=False):

        super(UnetBlockWithCmapAndNoise, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        self.nc = nc
        self.addcmap = addcmap
        self.addz = addz
        self.debug = debug
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        if outermost:
            input_nc += (nz + nc)
        else:
            if addcmap:
                input_nc += nc
            if addz:
                input_nc += nz
        if sn:
            downconv = nn.utils.spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias))
        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if sn:
                upconv = nn.utils.spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1))
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            #model = down + [submodule] + up
        elif innermost:
            if sn:
                upconv = nn.utils.spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias))
            else:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            #model = down + up
        else:
            if sn:
                upconv = nn.utils.spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias))
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, cmap=None, z=None):
        # print(x.size())
        if self.outermost:
            if self.nz > 0:
                z_img = z.view(z.size()[0], z.size()[1], 1, 1).expand(z.size()[0], z.size()[1], x.size()[2], x.size()[3])
                x_with_cmapz = torch.cat([x, cmap, z_img], 1)
            else:
                x_with_cmapz = torch.cat([x, cmap], 1)
            x1 = self.down(x_with_cmapz)
            x2 = self.submodule(x1, cmap, z)
            out = self.up(torch.cat([x2, x1], 1))
            if self.debug:
                nansum = torch.sum(torch.isnan(out))
                infsum = torch.sum(torch.isinf(out))
                if nansum > 0 or infsum > 0:
                    print('U-net outmost has %d nan and %d inf' % (nansum, infsum))
                assert nansum == 0 and infsum == 0
            return out
        else: # innermost and middle blocks
            x_with_cmapz = x
            if self.addcmap:
                cmap = F.interpolate(cmap, size=(x.size()[2], x.size()[3]), mode='bilinear')
                x_with_cmapz = torch.cat([x_with_cmapz, cmap], 1)
            if self.addz and self.nz > 0:
                z_img = z.view(z.size()[0], z.size()[1], 1, 1).expand(z.size()[0], z.size()[1], x.size()[2],
                                                                      x.size()[3])
                x_with_cmapz = torch.cat([x_with_cmapz, z_img], 1)
            if self.innermost:
                out = self.up(self.down(x_with_cmapz))
                if self.debug:
                    nansum = torch.sum(torch.isnan(out))
                    infsum = torch.sum(torch.isinf(out))
                    if nansum > 0 or infsum > 0:
                        print('U-net innermost has %d nan and %d inf' % (nansum, infsum))
                    assert nansum == 0 and infsum == 0
                return out
            else:
                x1 = self.down(x_with_cmapz)
                x2 = self.submodule(x1, cmap, z)
                out = self.up(torch.cat([x2, x1], 1))
                if self.debug:
                    nansum = torch.sum(torch.isnan(out))
                    infsum = torch.sum(torch.isinf(out))
                    if nansum > 0 or infsum > 0:
                        print('U-net mid has %d nan and %d inf' % (nansum, infsum))
                    assert nansum == 0 and infsum == 0
                return out


def define_enh(input_nc, output_nc, ngf, normType, netEnh='unetDown32', sn=False, debug=False):
        if netEnh == 'unetDown32':
            net = NormalUnet(input_nc=input_nc, output_nc=output_nc, blocksDown=[1,1,1,1,1], ngf=ngf,
                                normType=normType,  sn=sn, debug=debug)
        elif netEnh == 'unetDown64':
            net = NormalUnet(input_nc=input_nc, output_nc=output_nc, blocksDown=[1,1,1,1,1,1], ngf=ngf,
                                normType=normType,  sn=sn, debug=debug)
        elif netEnh == 'unet32':
            net = NormalUnet(input_nc=input_nc, output_nc=output_nc, blocksDown=[0, 0, 0, 0, 0], ngf=ngf,
                             normType=normType, sn=sn, debug=debug)
        elif netEnh == 'unet64':
            net = NormalUnet(input_nc=input_nc, output_nc=output_nc, blocksDown=[0, 0, 0, 0, 0, 0], ngf=ngf,
                             normType=normType, sn=sn, debug=debug)
        elif netEnh == 'resnetDown32':
            net = NormalResnet(input_nc=input_nc, output_nc=output_nc, ifdomw=True, ngf=ngf,
                             normType=normType, sn=sn, n_blocks=3)
        elif netEnh == 'resnetDown64':
            net = NormalResnet(input_nc=input_nc, output_nc=output_nc, ifdomw=True, ngf=ngf,
                             normType=normType, sn=sn, n_blocks=4)
        elif netEnh == 'resnet32':
            net = NormalResnet(input_nc=input_nc, output_nc=output_nc, ifdomw=False, ngf=ngf,
                             normType=normType, sn=sn, n_blocks=3)
        elif netEnh == 'resnet64':
            net = NormalResnet(input_nc=input_nc, output_nc=output_nc, ifdomw=False, ngf=ngf,
                             normType=normType, sn=sn, n_blocks=4)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netEnh)

        return net


class NormalResnet(nn.Module):
    def __init__(self, input_nc, output_nc, ifdomw=True, sn=False, ngf=64, normType='instance', n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(NormalResnet, self).__init__()
        norm_layer = get_norm_layer(normType)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if sn:
            model = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias))]
        else:
            model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias)]
        model += [
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if ifdomw:
                downconv = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
            else:
                downconv = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if sn:
                downconv = nn.utils.spectral_norm(downconv)
            model += [downconv,
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [NormalResBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, sn=sn, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if ifdomw:
                upconv =nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
            else:
                upconv = nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias)
            if sn:
                upconv = nn.utils.spectral_norm(upconv)

            model += [upconv,
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, bias=use_bias))]
        else:
            model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, bias=use_bias)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class NormalResBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, sn, use_bias):
        super(NormalResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, sn, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, sn, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if sn:
            conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        conv_block += [norm_layer(dim), nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if sn:
            conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out



class NormalUnet(nn.Module):
    def __init__(self, input_nc, output_nc, blocksDown = [1,1,1,1,1,1], ngf=64, normType='instance',  sn=False, debug=False):
        # blocksDown: len(blocksDown) = the number of unet blocks, and 1 means whether downsample the resolution, from left to right: inner to outer
        super(NormalUnet, self).__init__()
        assert len(blocksDown) >= 5
        norm_layer = get_norm_layer(normType)
        for i, ifdown in enumerate(blocksDown):
            if i==0:
                unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None,  DownRes = bool(ifdown), submodule=None, norm_layer=norm_layer,
                                       innermost=True, sn=sn, debug=debug)
            elif  i<= len(blocksDown) - 5:
                unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None,  DownRes = bool(ifdown), submodule=unet_block, norm_layer=norm_layer,
                                       sn=sn, debug=debug)
            elif i < len(blocksDown) - 1:
                multi = 2**(len(blocksDown) - i - 1)
                unet_block = UnetBlock(int(ngf*(multi/2)), ngf*multi, input_nc=None, DownRes = bool(ifdown), submodule=unet_block, norm_layer=norm_layer, sn=sn,
                                   debug=debug)
            elif i == len(blocksDown) - 1:
                self.model = UnetBlock(output_nc, ngf, input_nc=input_nc, DownRes = bool(ifdown), submodule=unet_block, outermost=True,
                                       norm_layer=norm_layer, sn=sn, debug=debug)

    def forward(self, x):
        return self.model(x)


class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, DownRes = True,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, sn=False, debug=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.debug = debug
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        if not DownRes:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        if sn:
            downconv = nn.utils.spectral_norm(downconv)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if not DownRes:
                upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                            kernel_size=3, stride=1,
                                            padding=1)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            if sn:
                upconv = nn.utils.spectral_norm(upconv)

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            #model = down + [submodule] + up
        elif innermost:
            if not DownRes:
                upconv = nn.Conv2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            else:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if sn:
                upconv = nn.utils.spectral_norm(upconv)

            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            #model = down + up
        else:
            if not DownRes:
                upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if sn:
                upconv = nn.utils.spectral_norm(upconv)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x):
        # print(x.size())
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1)
            out = self.up(torch.cat([x2, x1], 1))
            if self.debug:
                nansum = torch.sum(torch.isnan(out))
                infsum = torch.sum(torch.isinf(out))
                if nansum > 0 or infsum > 0:
                    print('Enhancement U-net outmost has %d nan and %d inf' % (nansum, infsum))
            return out
        else: # innermost and middle blocks
            if self.innermost:
                out = self.up(self.down(x))
                if self.debug:
                    nansum = torch.sum(torch.isnan(out))
                    infsum = torch.sum(torch.isinf(out))
                    if nansum > 0 or infsum > 0:
                        print('Enhancement U-net innermost has %d nan and %d inf' % (nansum, infsum))
                return out
            else:
                x1 = self.down(x)
                x2 = self.submodule(x1)
                out = self.up(torch.cat([x2, x1], 1))
                if self.debug:
                    nansum = torch.sum(torch.isnan(out))
                    infsum = torch.sum(torch.isinf(out))
                    if nansum > 0 or infsum > 0:
                        print('Enhancement U-net mid has %d nan and %d inf' % (nansum, infsum))
                return out
####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'linear':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler


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


def init_weights(net, init_type='xavier', init_gain=1):
    def init_func(m):  # define the initialization function
      classname = m.__class__.__name__
      if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif classname.find(
              'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='xavier', init_gain=1, device='cpu', gpu_ids=[]):
  if len(gpu_ids) > 0:
    assert (torch.cuda.is_available())
    net.to(device)
    net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
  init_weights(net, init_type, init_gain=init_gain)
  return net


# def normSum1(mat, eps=1e-7):  # tensor [batch_size, channels, image_height, image_width] normalize each fea map
#       mat_shape = mat.size
#       if len(mat_shape) == 2:
#           tempsum = torch.sum(mat) + eps
#       elif len(mat_shape) == 3:
#           tempsum = torch.sum(mat, dim=(1, 2), keepdim=True) + eps
#       elif len(mat_shape) == 4:
#           tempsum = torch.sum(mat, dim=(2, 3), keepdim=True) + eps
#       return mat / tempsum

def NormAffine(mat, eps=1e-7,
                 method='sum'):  # tensor [batch_size, channels, image_height, image_width] normalize each fea map;
    matdim = len(mat.size())
    if method == 'sum':
      tempsum = torch.sum(mat, dim=(matdim - 1, matdim - 2), keepdim=True) + eps
      out = mat / tempsum
    elif method == 'one':
      (tempmin,_) = torch.min(mat, dim=matdim - 1, keepdim=True)
      (tempmin,_) = torch.min(tempmin, dim=matdim - 2, keepdim=True)
      tempmat = mat - tempmin
      (tempmax,_) = torch.max(tempmat, dim=matdim - 1, keepdim=True)
      (tempmax,_) = torch.max(tempmax, dim=matdim - 2, keepdim=True)
      tempmax = tempmax + eps
      out = tempmat / tempmax
    else:
      raise NotImplementedError('Map method [%s] is not implemented' % method)
    return out


def SigmoidBlur(mat, alpha=10, beta=0.2):
    newmat = 1.0 / (1 + torch.exp(-alpha * (mat - beta)))
    newmat.where(newmat>beta, torch.zeros_like(newmat))
    #newmat2 = torch.clamp(newmat - beta, min=0) + beta
    return newmat

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, sn=False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU, debug=False, id=None, idb=None):
        super(Bottleneck, self).__init__()
        self.debug = debug
        self.id = id
        self.idb = idb
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
        if self.debug:
            nansum = torch.sum(torch.isnan(out))
            infsum = torch.sum(torch.isinf(out))
            if nansum > 0 or infsum > 0:
                print('%d-th bottleneck of %d-th block has %d nan and %d inf' % (self.id, self.idb, nansum, infsum))
            assert nansum == 0 and infsum == 0

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


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, sn=False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU, debug=False,id=None):
        super(Transition, self).__init__()
        self.debug = debug
        self.id = id
        if sn:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.utils.spectral_norm(nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)),
                      nn.AvgPool2d(2)
                      ]
        else:
            layers = [norm_layer(nChannels),
                      act_layer(),
                      nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False),
                      nn.AvgPool2d(2)
                      ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        if self.debug:
            nansum = torch.sum(torch.isnan(out))
            infsum = torch.sum(torch.isinf(out))
            if nansum > 0 or infsum > 0:
                print('Densenet %d-th trans block has %d nan and %d inf' % (self.id, nansum, infsum))
            assert nansum == 0 and infsum == 0
        return out


class DenseBlock(nn.Module):
  def __init__(self, nChannels, growthRate, nLayers, reduction, bottleneck=True, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU, sn=False, debug=False, id=None):
    super(DenseBlock, self).__init__()
    layers = []
    self.id = id
    self.debug = debug
    for i in range(int(nLayers)):
      if bottleneck:
        layers += [Bottleneck(nChannels, growthRate, sn, norm_layer, act_layer, debug, i, id)]
      else:
        layers += [SingleLayer(nChannels, growthRate, sn, norm_layer, act_layer)]
      nChannels += growthRate
    nOutChannels = int(math.floor(nChannels * reduction))
    layers += [Transition(nChannels, nOutChannels, sn, norm_layer, act_layer, debug, id)]
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    out = self.model(x)
    if self.debug:
        nansum = torch.sum(torch.isnan(out))
        infsum = torch.sum(torch.isinf(out))
        if nansum > 0 or infsum > 0:
            print('Densenet %d-th block has %d nan and %d inf' % (self.id, nansum, infsum))
        assert nansum == 0 and infsum == 0
    return out


class DecoderBlock(nn.Module):
  def __init__(self, nChannels, nOutChannels, deConv = False, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU):
    super(DecoderBlock, self).__init__()
    interChannels = int(math.ceil(nChannels*math.sqrt(nOutChannels/nChannels)))
    if deConv:
      #output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
      layers = [norm_layer(nChannels),
                act_layer(),
                nn.ConvTranspose2d(nChannels, interChannels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
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

class NormActConv(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, act_layer, norm_layer, sn=False, padding_type = 'zero'):
    super(NormActConv, self).__init__()
    layers = []
    if padding_type == 'reflect':
        layers += [norm_layer(n_in),
                  act_layer(),
                  nn.ReflectionPad2d(padding)]
        p = 0
    elif padding_type == 'replicate':
        layers += [norm_layer(n_in),
                  act_layer(),
                nn.ReplicationPad2d(padding)]
        p = 0
    elif padding_type == 'zero':
        layers += [norm_layer(n_in),
                  act_layer()]
        p = padding
    else:
        raise NotImplementedError(
            'padding [%s] is not implemented' % padding_type)
    if sn:
        layers += [nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=p))]
    else:
        layers += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=p)]
    self.model = nn.Sequential(*layers)
  def forward(self, x):
    return self.model(x)


class ASPP(nn.Module):
      def __init__(self, in_channel=512, depth=256, scales=(0, 1, 4, 8, 12), sn=False):
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
                    layers = [nn.utils.spectral_norm(nn.Conv2d(in_channel, depth, 3, 1, dilation=dilate_rate, padding=dilate_rate))]
                  else:
                    layers = [nn.Conv2d(in_channel, depth, 3, 1, dilation=dilate_rate, padding=dilate_rate)]
                  setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))
          # if sn:
          #     self.conv_1x1_output = nn.utils.spectral_norm(nn.Conv2d(depth * len(scales), depth, 1, 1))
          # else:
          #     self.conv_1x1_output = nn.Conv2d(depth * len(scales), depth, 1, 1)
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
#-------------------------- Losses --------------------------
####################################################################
def KL_loss(out, gt, eps=1e-7):
  assert out.size() == gt.size()
  out =  NormAffine(out, eps=1e-7, method='sum')
  gt = NormAffine(gt, eps=1e-7, method='sum')
  loss = torch.sum(gt * torch.log(eps + gt / (out + eps)))
  loss = loss / out.size()[0]
  return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real, bias):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label + bias
        else:
            target_tensor = self.fake_label + bias
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real, lossweight, bias=0):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real, bias)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            loss = lossweight * loss
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss, all_losses


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

######################################################Pix2pix##################################################################




class UnetGeneratorPIX(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, normType='instance', use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGeneratorPIX, self).__init__()
        norm_layer = get_norm_layer(normType)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

