import torch
from torchvision import transforms, utils
from PIL import Image
import argparse
import os
import glob
from metrics import auc_judd, nss

import torch.nn.functional as F

# Read the data
toTensor = transforms.Compose([
    transforms.Resize(720),
    transforms.ToTensor(),
])


def read(imgDir: str) -> torch.Tensor:
    x = toTensor(Image.open(imgDir))
    return x


def calAll(path1: str, path2: str, name: str):
    aucAll = 0
    nssAll = 0
    img_list1 = glob.glob(f"{path1}/*.jpg")
    tbs = len(img_list1)
    with torch.no_grad():
        for i, path in enumerate(img_list1):
            # imgList1 = []
            # imgList2 = []
            # print(f"[{i:05}|{tbs}] | {1}", end="\r")
            cn = path.split("/")[-1].split(".")[0]
            # breakpoint()
            img1 = read(path)
            img2 = read(f"{path2}/{cn}_fixMap.jpg")
            # if i in allwrong:
            #     cnt += 1
            #     print(f"weird case {i} jump {cnt=}")
            #     continue
            # breakpoint()
            # imgList1.append(img1)
            # imgList2.append(img2)
            # img1 = torch.stack(imgList1).cuda()
            # img2 = torch.stack(imgList2).cuda()

            singauc = auc_judd(img1, img2)
            singnss = nss(img1, img2)
            print(
                f"[{i:05}|{tbs}]\t\t\t\tauc path2 {(singauc):.4f} | nss {(singnss):.4f}", end="\r")
            with open(f"record/{name}/record", "a") as f:
                f.write(f"[{i:05}|{tbs}]\t\t\t\tauc path2 {(singauc):.4f} | nss {(singnss):.4f}\n")

            aucAll += singauc
            nssAll += singnss
        print(f"{path1} {(aucAll / tbs):.4f} | {(nssAll / tbs):.4f}")
        with open(f"record/all_record.txt", "a") as f:
            f.write(f"{path1} | auc nss | {(aucAll / tbs):.4f} | {(nssAll / tbs):.4f}\n")
        # print(f"[INFO]: id ORI {(idCompAll / tbs):.4f} id {(idAll / tbs):.4f} psnr ORI {(qpsnrAll / tbs):.4f} psnr {(psnrCompAll / tbs):.4f} ")


parser = argparse.ArgumentParser()
parser.add_argument('--path1', type=str)
parser.add_argument('--path2', type=str, default="/mnt/hdd1/yifei/DATA/ECdata/ALLFIXATIONMAPS")
args = parser.parse_args()

PATH1 = args.path1
PATH2 = args.path2
dirlis = args.path1.split("/")
name = f"{dirlis[-3]}_{dirlis[-2]}_{dirlis[-1]}"
os.makedirs(f"record/{name}", exist_ok=True)

calAll(PATH1, PATH2, name)
