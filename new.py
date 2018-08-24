from PIL import Image
import cv2
import glob as gb

im_path=gb.glob(".\\celldata\\train\\*.tif")
#print(len(im_path))

for path in im_path:
    im=cv2.imread(path)
    #cv2.imshow('img',im)
    #cv2.waitKey(1000)

print(cv2.imread(im_path[0]))