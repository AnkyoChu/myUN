import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import glob as gb

#device configuratiom
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dataset
#data size 512*512*3
im_path=gb.glob(".\\celldata\\train\\*.tif")
#print(len(im_path))
for path in im_path:
    im=cv2.imread(path)
    #cv2.imshow('img',im)
    #cv2.waitKey(1000)

#print(im[0].size,im[0])