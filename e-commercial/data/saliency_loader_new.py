import os
import os.path
import torch, csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import glob
import numpy as np
random.seed(1)
globaltest = []

def ecommercedata(data_path, img_nums, is_train):
    global globaltest
    nums = [i for i in range(1, img_nums + 1)]
    if not globaltest:
        globaltest = random.sample(nums, img_nums // 10)
    test = globaltest
    train = list(set(nums) - set(test))
    if is_train:
        return EcommerceDataset(data_path=data_path, img_nums=(img_nums - (img_nums // 10)),
                                lis=train)
    else:
        return EcommerceDataset(data_path=data_path, img_nums=(img_nums // 10),
                                lis=test)


class EcommerceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, img_nums, transform=None, lis=[]):
        """
        Args:
            data_path (string): Path to the imgs file with saliency.
            img_nums (int): Total number of images to index.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.transform = transform
        self.data_len = img_nums
        self.lis = lis

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx", idx, "length", len(self.lis))

        img_name = 'ALLSTIMULI/' + str(self.lis[idx]) + '.jpg'
        saliency_name = 'ALLFIXATIONMAPS/' + str(self.lis[idx]) + '_fixMap.jpg'
        ocr_aff_name = 'OCR/affinity/' + str(self.lis[idx]) + '.csv'
        ocr_reg_name = 'OCR/region/' + str(self.lis[idx]) + '.csv'
        img_file = os.path.join(self.root_dir, img_name)
        saliency_file = os.path.join(self.root_dir, saliency_name)
        ocr_aff_file = os.path.join(self.root_dir, ocr_aff_name)
        ocr_reg_file = os.path.join(self.root_dir, ocr_reg_name)

        # images
        img = Image.open(img_file)
        # img = np.array(img, dtype=np.float32)  # h, w, c
        torch_img = transforms.functional.to_tensor(img)
        torch_img = transforms.Resize(896)(torch_img)
        # saliency
        saliency = Image.open(saliency_file)
        # saliency = np.array(saliency, dtype=np.float32)  # h, w, c
        torch_saliency = transforms.functional.to_tensor(saliency)
        torch_saliency = transforms.Resize(896)(torch_saliency)
        # ocr
        csv_aff_content = np.loadtxt(open(ocr_aff_file, "rb"), delimiter=",")
        csv_reg_content = np.loadtxt(open(ocr_reg_file, "rb"), delimiter=",")
        # to make pytorch happy in transforms
        csv_aff_image = np.expand_dims(csv_aff_content, axis=0)
        csv_reg_image = np.expand_dims(csv_reg_content, axis=0)
        torch_aff = torch.from_numpy(csv_aff_image).float()
        torch_reg = torch.from_numpy(csv_reg_image).float()
        gh_label = transforms.Resize(896)(torch_aff)
        gah_label = transforms.Resize(896)(torch_reg)
        # print('The sizes of gh_label and gah_label are :', gh_label.size(), gah_label.size())
        if self.transform:
            raise NotImplementedError("Not support any transform by far!")
            # sample = self.transform(sample)

        return torch_img, torch_saliency
