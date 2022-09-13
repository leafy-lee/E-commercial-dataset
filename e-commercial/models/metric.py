import numpy as np
import torch


def calCC(gtsAnns, resAnns, is_train):
    if is_train:
        gtsAnn = gtsAnns[0, ...].detach().clone()
        gtsAnn = torch.cat([gtsAnn, gtsAnn, gtsAnn], 0)
        gtsAnn = gtsAnn.cpu().float().detach().numpy()
        resAnn = resAnns[0, ...].detach().clone()
        resAnn = torch.cat([resAnn, resAnn, resAnn], 0)
        resAnn = resAnn.cpu().float().detach().numpy()
        fixationMap = gtsAnn - np.mean(gtsAnn)
        if np.max(fixationMap) > 0:
            fixationMap = fixationMap / np.std(fixationMap)
        salMap = resAnn - np.mean(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.std(salMap)
        return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]
    else:
        cc = 0
        for idx, gtsAnn in enumerate(gtsAnns):
            gtsAnn = gtsAnn.detach().clone()
            gtsAnn = torch.cat([gtsAnn, gtsAnn, gtsAnn], 0)
            gtsAnn = gtsAnn.cpu().float().detach().numpy()
            resAnn = resAnns[idx].detach().clone()
            resAnn = torch.cat([resAnn, resAnn, resAnn], 0)
            resAnn = resAnn.cpu().float().detach().numpy()
            fixationMap = gtsAnn - np.mean(gtsAnn)
            if np.max(fixationMap) > 0:
                fixationMap = fixationMap / np.std(fixationMap)
            salMap = resAnn - np.mean(resAnn)
            if np.max(salMap) > 0:
                salMap = salMap / np.std(salMap)
            cc += np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]
        return cc / gtsAnns.size()[0]


def calKL(gtsAnns, resAnns, is_train, eps=1e-7):
    if is_train:
        gtsAnn = gtsAnns[0, ...].detach().clone()
        gtsAnn = torch.cat([gtsAnn, gtsAnn, gtsAnn], 0)
        gtsAnn = gtsAnn.cpu().float().detach().numpy()
        resAnn = resAnns[0, ...].detach().clone()
        resAnn = torch.cat([resAnn, resAnn, resAnn], 0)
        resAnn = resAnn.cpu().float().detach().numpy()
        if np.sum(gtsAnn) > 0:
            gtsAnn = gtsAnn / np.sum(gtsAnn)
        if np.sum(resAnn) > 0:
            resAnn = resAnn / np.sum(resAnn)
        return np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))
    else:
        kl = 0
        for idx, gtsAnn in enumerate(gtsAnns):
            gtsAnn = gtsAnn.detach().clone()
            gtsAnn = torch.cat([gtsAnn, gtsAnn, gtsAnn], 0)
            gtsAnn = gtsAnn.cpu().float().detach().numpy()
            resAnn = resAnns[idx].detach().clone()
            resAnn = torch.cat([resAnn, resAnn, resAnn], 0)
            resAnn = resAnn.cpu().float().detach().numpy()
            if np.sum(gtsAnn) > 0:
                gtsAnn = gtsAnn / np.sum(gtsAnn)
            if np.sum(resAnn) > 0:
                resAnn = resAnn / np.sum(resAnn)
            kl += np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))
        return kl / gtsAnns.size()[0]