from .inception_score.inception_score import inception_score
from .fid.fid import calculate_fid_given_paths
from .ssim.ssim import ssim
from .psnr.psnr import psnr
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
"""
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
def calculate_fid_given_paths(paths, batch_size, cuda, dims):
def ssim(img1, img2, window_size = 11, size_average = True):
"""
SIZE = (256,256)
_transforms_fun=transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
def _inception_score(path, cuda=False, batch_size=1, resize=True, splits=1):
    imgs = []
    for file in os.listdir(path):
        if file.endswith("png"):
            img = Image.open(os.path.join(path, file)).convert("RGB")
            #print(np.array(img).shape)
            imgs.append(_transforms_fun(img))
    imgs = torch.stack(imgs)
    #print(imgs.size())
    return inception_score(imgs, cuda, batch_size, resize, splits)

def _fid(paths, batch_size=10, cuda=False, dims=2048):
    return calculate_fid_given_paths(paths, batch_size, cuda, dims)

def _ssim(paths, window_size=11, size_average=True):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    batch_size = 100
    j = 0
    total = 0
    ssim_score = 0
    for file in os.listdir(path1):
        if file.endswith("png"):
            img1 = Image.open(os.path.join(path1, file)).convert("RGB")
            img2 = Image.open(os.path.join(path2, file)).convert("RGB")

            imgs1.append(_transforms_fun(img1))
            imgs2.append(_transforms_fun(img2))
            j = j + 1
            total = total + 1
        if j == batch_size - 1:
            imgs1 = torch.stack(imgs1)
            imgs2 = torch.stack(imgs2)
            ssim_score = ssim_score + batch_size * ssim(imgs1, imgs2, window_size = 11, size_average = True)
            imgs1, imgs2 = [], []
            j = 0
    if j != 0:
        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)
        ssim_score = ssim_score +  (j+1) * ssim(imgs1, imgs2, window_size = 11, size_average = True)

    return ssim_score / total

def _psnr(paths):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    psnr_value = 0
    num = 1
    for file in os.listdir(path1):
        if file.endswith("png"):
            img1 = Image.open(os.path.join(path1, file)).convert("RGB")
            img2 = Image.open(os.path.join(path2, file)).convert("RGB")
            psnr_value = psnr_value + psnr(cv2.resize(np.array(img1),SIZE), cv2.resize(np.array(img2), SIZE))
            num = num + 1

    return psnr_value / num

def _meanl1(paths):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    total_error = 0
    num = 1
    for file in os.listdir(path1):
        if file.endswith("png"):
            img1 = Image.open(os.path.join(path1, file)).convert("RGB")
            img2 = Image.open(os.path.join(path2, file)).convert("RGB")

            l1_error = np.mean(np.abs(cv2.resize(np.array(img1),SIZE)-cv2.resize(np.array(img2), SIZE)))
            #print(np.array(img1).shape, l1_error,np.sum(np.abs(cv2.resize(np.array(img1),SIZE)-cv2.resize(np.array(img2), SIZE)))/256/256/3)
            total_error = total_error + l1_error
            num = num + 1

    return total_error / num

metrics = {"is":_inception_score, "fid":_fid, "ssim":_ssim, "psnr":_psnr, "meanl1":_meanl1}
