import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################
# -------------------------- Losses --------------------------
####################################################################


import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        # print(loss_label.shape, pre_loss.shape)
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
            #sum_loss += loss/average_number

        return sum_loss



    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        mask = mask.squeeze(1)

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss2 = loss_fn(p_gah, gah_label)
        # print("loss1.shape, mask.shape", loss1.shape, mask.shape)
        # print("loss2.shape, mask.shape", loss2.shape, mask.shape)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)
        # print("loss shape", loss_g.shape, loss_a.shape, gah_label.shape, gh_label.shape)

        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]
        

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


def KL_loss(out, gt):
    assert out.size() == gt.size()
    out = NormAffine(out, eps=1e-7, method='sum')
    gt = NormAffine(gt, eps=1e-7, method='sum')
    loss = torch.sum(gt * torch.log(1e-7 + gt / (out + 1e-7)))
    loss = loss / out.size()[0]
    return loss
