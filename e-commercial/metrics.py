import cv2
import torch
import numpy as np
from skimage.transform import resize
from numpy import random
from functools import partial

def loss_func(pre, gt_sal, gt_fix, args):

    kl = kld_loss(pre, gt_sal)
    cc1 = cc(pre, gt_sal)
    nss1 = nss(pre, gt_fix)

    return kl, cc1, nss1


def kld_loss(s_map, gt):  # 1,1280,720    1,1280,720
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    # 以下相当于对feature的每一个元素做了归一化
    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)  # batch_size个和
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)  # 1,1280,720,value=sum_s_map
    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)  # 1,1280,720
    gt = gt / (expand_gt * 1.0)  # 1,1280,720

    s_map = s_map.view(batch_size, -1)  # 1,921600
    gt = gt.view(batch_size, -1)  # B,921600

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt / (s_map + eps))  # B,921600

    return torch.mean(torch.sum(result, 1))  # 返回的是batch个kl的平均值


def cc(s_map, gt):  # 1,1280,720    1,1280,720
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa * bb)))


def nss(s_map, gt):
# def cal_nss_wsj(s_map, gt):
    # print(">> s_map ", s_map, s_map.size())
    # print(">> gt ", gt, gt.size())

    # if s_map.size() != gt.size():
    #     s_map = s_map.cpu().squeeze(0).numpy()
    #     s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
    #     s_map = s_map.cuda()
    #     gt = gt.cuda()
    # print(s_map.size(), gt.size())
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.view(batch_size, -1), 1)
    return torch.mean(s_map / count)


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


def similarity(s_map, gt):
    """
    For single image metric
    Size of Image - WxH or 1xWxH
    gt is ground truth saliency map
    """
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    s_map = normalize_map(s_map)
    gt = normalize_map(gt)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    return torch.mean(torch.sum(torch.min(s_map, gt), 1))


def auc_judd(saliencyMap, fixationMap, jitter=True, toPlot=False, normalize=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    #       ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if saliencyMap.size() != fixationMap.size():
        saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
        saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(
            0)
        # saliencyMap = saliencyMap.cuda()
        # fixationMap = fixationMap.cuda()
    if len(saliencyMap.size()) == 3:
        saliencyMap = saliencyMap[0, :, :]
        fixationMap = fixationMap[0, :, :]
    # saliencyMap = saliencyMap.numpy()
    # fixationMap = fixationMap.numpy()
    saliencyMap = saliencyMap
    fixationMap = fixationMap
    if normalize:
        saliencyMap = normalize_map(saliencyMap)

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap.cpu() + np.random.random(np.shape(saliencyMap.cpu())) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten().cpu()
    F = fixationMap.flatten().cpu()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    score = score
    return score


def auc_judd_npy(saliencyMap, fixationMap, jitter=True, toPlot=False, normalize=False):

    # If there are no fixations to predict, return NaN
    # if saliencyMap.size() != fixationMap.size():
    #     saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
    #     saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(
    #         0)
        # saliencyMap = saliencyMap.cuda()
        # fixationMap = fixationMap.cuda()
    # if len(saliencyMap.size()) == 3:
    #     saliencyMap = saliencyMap[0, :, :]
    #     fixationMap = fixationMap[0, :, :]
    # saliencyMap = saliencyMap.numpy()
    # fixationMap = fixationMap.numpy()
    saliencyMap = saliencyMap
    fixationMap = fixationMap
    if normalize:
        saliencyMap = normalize_map(saliencyMap)

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten() #.cpu()
    F = fixationMap.flatten() #.cpu()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)


    score = score
    return score


def auc_shuff(s_map, gt, other_map, splits=100, stepsize=0.1):
    if len(s_map.size()) == 3:
        s_map = s_map[0, :, :]
        gt = gt[0, :, :]
        other_map = other_map[0, :, :]

    s_map = s_map.cpu().numpy()
    s_map = (s_map-np.min(s_map))/(np.max(s_map)-np.min(s_map))
    gt = gt.cpu().numpy()
    other_map = other_map.cpu().numpy()

    num_fixations = np.sum(gt)

    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, int(k / s_map.shape[0])])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


# 来源：SalEMA
def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    rand_sampler : callable
        S_rand = rand_sampler(S, F, n_rep, n_fix)
        Sample the saliency map at random locations to estimate false positive.
        Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map.cpu(), copy=False)
    fixation_map = np.array(fixation_map.cpu(), copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # Normalize saliency map to have values between [0,1]
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc) # Average across random splits


def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ON OTHER IMAGES),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    other_map : binary matrix, same shape as fixation_map
        A binary fixation map (like fixation_map) by taking the union of fixations from M other random images
        (Borji uses M=10).
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    Returns
    -------
    AUC : float, between [0,1]
    '''
    other_map = np.array(other_map.cpu(), copy=False) > 0.5

    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')

    # For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
    def sample_other(other, S, F, n_rep, n_fix):
        fixated = np.nonzero(other)[0]
        indexer = list(map(lambda x: random.permutation(x)[:n_fix], np.tile(range(len(fixated)), [n_rep, 1])))
        r = fixated[np.transpose(indexer)]
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
        return S_rand
    return AUC_Borji(saliency_map, fixation_map, n_rep, step_size, partial(sample_other, other_map.ravel()))



