#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 00:00:11 2019

@author: yuanzhu
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

    
###1.a read a rgb image and display it
img=cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
#print(img)
cv2.imshow("origin",img)
print(img.shape)

###1.b Convert the RGB image into Lab colour system
lab_image = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
l_channel,a_channel,b_channel = cv2.split(lab_image)
print(l_channel.shape)
cv2.imshow("l_channel",l_channel)
cv2.imshow("a_channel",a_channel)
cv2.imshow("b_channel",b_channel) 
#plt.imshow(l_channel, cmap='gray')

###1.c
#edge detection
#GassuabBlur smooth picture and remove noise
blurred = cv2.GaussianBlur(l_channel, (3, 3), 0)
Gx=np.array([
     [-1,0,1],
     [-2,0,2],
     [-1,0,1]
])#the second row is 0;right - left
Gy=np.array([
     [-1,-2,-1],
     [0,0,0],
     [1,2,1]
])
#convolve x,y
PX = cv2.filter2D(blurred,-1,Gx)
PY = cv2.filter2D(blurred,-1,Gy)
absX = cv2.convertScaleAbs(PX)
absY = cv2.convertScaleAbs(PY)
cv2.imshow("px", PX)
cv2.imshow("py", PY)

#check this step, edge detection using Laplacian
result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imshow("edge_det", result)#compare with laplacian result 
laplacian = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(laplacian)
cv2.imshow("edge_dst", dst)

###1.d Compute a 2D histogram
hist = cv2.calcHist([lab_image], [1,2], None,[256, 256], [0,256,0,256])#[-128,127]
cv2.imshow("hist", hist)#grey imagine

###1.e back projection, try to find the model in origional image
part_img=img[100:200,100:200]#the part area(hat) going to find
cv2.imshow("part_img", part_img)
lab_partimage = cv2.cvtColor(part_img,cv2.COLOR_BGR2Lab)
hist_partimage= cv2.calcHist([lab_partimage], [1,2], None,[256, 256], [0,256,0,256])#[-128,127]
mask=cv2.calcBackProject([lab_image],[1,2],hist_partimage,[0,256,0,256],1)
cv2.imshow("mask", mask)
#check the area, black area is not the model area, white area is similar area

###1.f histogram equalization(l)
plt.hist(l_channel.ravel(),256)
plt.show()
equ=cv2.equalizeHist(l_channel)
#new=cv2.merge((equ,a_channel,b_channel))
plt.hist(equ.ravel(),256)
plt.show()
cv2.imshow("before_equ", l_channel)
cv2.imshow("after_equ", equ)
#the equ img has more contrast compare to the origion one


cv2.waitKey(0)
cv2.destroyAllWindows()