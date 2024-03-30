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

def ecommercedata(data_path, img_nums, is_train, logger):
    global globaltest
    nums = [i for i in range(1, img_nums + 1)]
    if not globaltest:
        globaltest = random.sample(nums, img_nums // 10)
    test = globaltest
    logger.info(f"now globaltest is {globaltest}")
    train = list(set(nums) - set(test))
    if is_train:
        print(f"using {(img_nums - (img_nums // 10))} images for train")
        return EcommerceDataset(data_path=data_path, img_nums=(img_nums - (img_nums // 10)),
                                lis=train)
    else:
        print(f"using {(img_nums // 10)} images for test")
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
        gh_label = transforms.Resize(224)(torch_aff)
        gah_label = transforms.Resize(224)(torch_reg)
        # print('The sizes of gh_label and gah_label are :', gh_label.size(), gah_label.size())
        if self.transform:
            raise NotImplementedError("Not support any transform by far!")
            # sample = self.transform(sample)

        return self.lis[idx], torch_img, torch_saliency, \
               {'gh_label': gh_label, 'gah_label': gah_label, 'mask': torch.ones_like(gah_label)}


class finetunedata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the imgs file with saliency.
            img_nums (int): Total number of images to index.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.transform = transform
        path = os.path.join(self.root_dir, 'stimuli/')
        path2 = os.path.join(self.root_dir, 'fixation/')
        print("using imgs in %s"%path)
        imgs = [f for f in glob.glob(path+"*.jpg")]
        gts = [f for f in glob.glob(path2+"*.jpg")]
        print("imgs are like %s"%imgs[0])
        self.data_len = len(imgs)
        print("using %d imgs in training"%self.data_len)
        self.lis = imgs
        self.gts = gts

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_file = str(self.lis[idx])
        saliency_file =  str(self.gts[idx])

        # images
        img = Image.open(img_file)
        # img = np.array(img, dtype=np.float32)  # h, w, c
        torch_img = transforms.functional.to_tensor(img)
        torch_img = transforms.Resize((896,896))(torch_img)
        # saliency
        saliency = Image.open(saliency_file)
        # saliency = np.array(saliency, dtype=np.float32)  # h, w, c
        torch_saliency = transforms.functional.to_tensor(saliency)
        torch_saliency = transforms.Resize((896,896))(torch_saliency)
        # print("traing", torch_img.shape, torch_saliency.shape)
        if torch_img.size()[0] == 1:
            torch_img = torch_img.expand([3,896,896])
        if self.transform:
            raise NotImplementedError("Not support any transform by far!")
            # sample = self.transform(sample)

        return str(self.lis[idx]), torch_img, torch_saliency

class folderimagedata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the imgs file with saliency.
            img_nums (int): Total number of images to index.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.transform = transform
        print("using imgs in %s"%self.root_dir)
        supported = ["jpg", "jpeg"]
        imgs = []
        for i in supported:
            types = [f for f in glob.glob(self.root_dir+"*."+i)]
            imgs += types
        print("imgs are like %s"%imgs[0])
        self.data_len = len(imgs)
        print("using %d imgs in validate folder"%self.data_len)
        self.lis = imgs

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_file = str(self.lis[idx])

        # images
        img = Image.open(img_file)
        # img = np.array(img, dtype=np.float32)  # h, w, c
        torch_img = transforms.functional.to_tensor(img)
        torch_img = transforms.Resize((896,896))(torch_img)
        print(torch_img.size())
        if torch_img.size()[0] == 1:
            torch_img = torch_img.expand([3,896,896])
        if self.transform:
            raise NotImplementedError("Not support any transform by far!")
            # sample = self.transform(sample)

        return str(self.lis[idx]), torch_img, torch_img[:1].unsqueeze(0)
