#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:24:27 2019

@author: yuanzhu
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def show_img(row784):

    plt.figure(figsize=(3,3))
    im =row784.reshape(28, 28)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.close()
    
##read mnist images-6000rows 684cols
images,labels=load_mnist("mnist")


###a.  Compute the mean image and principal components for a set of images. 
      #a,Display the mean image and the first 2principal components (associated with the highest eigenvalues)

##select 1000 '5' pictures to train
trainset = np.zeros((1000,784),dtype=np.uint8)
test_img = np.zeros((1,784),dtype=np.uint8)

for i in range(1000):
    trainset[i] = images[labels == 5][i]
test_img[0]=images[labels == 5][1000]
 

#Compute the mean image
average = np.mean(trainset, axis=0)
#print(trainset.shape,average.shape)  #(1000,784) (784,)
print("Show the Mean Image:")
show_img(average)

#Compute the first 2principal components
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)#top 2 components
low = pca.fit_transform(trainset) #(1000,2)
eigenfaces = pca.components_  #(2,784)
#print(eigenfaces.shape)
print("first 2 principal components and images:\n",low)
show_img(eigenfaces[0]) #the image depend on first principal component
show_img(eigenfaces[1]) #the image depend on second principal component
                        #print(pca.n_components)
                        #print(pca.explained_variance_ratio_)#contribute percentage




###b. Compute and display the reconstructions of a test image using the mean image and with
        #principal components associated with the p highest eigenvalues (e.g. Fig 10.12) with
        #p=10 and p=50.
from sklearn import decomposition
pca = decomposition.PCA(n_components=10)#p=10
low = pca.fit_transform(trainset) #(1000,10)
eigenfaces = pca.components_  #(10,784)
eigenfaces=np.mat(eigenfaces)
c=(test_img-average)*eigenfaces.T*eigenfaces+average
print("Show the p=10 Image:")
show_img(c)#reconstruction image

pca = decomposition.PCA(n_components=50)#p=50
low = pca.fit_transform(trainset) #(1000,50)
eigenfaces = pca.components_  #(50,784)
eigenfaces=np.mat(eigenfaces)
c=(test_img-average)*eigenfaces.T*eigenfaces+average
print("Show the p=50 Image:")
show_img(c)#reconstruction image


###c. Compute and display a DFFS (distance-from feature-space)
        #and SSD (sum-of-square-differences) heat maps

def error(yhat,label):
    yhat = np.array(yhat)
    label = np.array(label)
    error_sum = ((yhat - label)**2).sum()
    return error_sum

def error_DFFs(yhat,label):
    yhat = np.array(yhat)
    label = np.array(label)
    error_sum = (yhat**2).sum()-(label**2).sum()
    return error_sum
#create a combined imagine(a big test image)
trainset2 = np.zeros((100,784),dtype=np.uint8)
    #get 10 images with the same number 0-9
k=0
for i in range(10):
    for j in range(10):
        trainset2[k] = images[labels == i][j]
        k=k+1

    #combine the image in col
def recPic(k):
    a=trainset2[k].reshape(28, 28)
    for i in range(9):
        b=trainset2[k+i+1].reshape(28, 28)
        a=np.vstack((a,b))
    return a
cc = np.zeros((10,280,28),dtype=np.uint8)
for i in range(10):
    cc[i]=recPic(10*i)

    #combine the image in row
dd=cc[0]
for i in range(9):
    dd=np.hstack((dd,cc[i+1]))
print("Show the big test Image:")
plt.imshow(dd, cmap='gray')#(280, 280)

#Compute and display a ssd & DFFS

def compu(x,y):#(x,y) position of pixel in test image
    hh = np.zeros((1,784),dtype=np.uint8)
    k=0
    for i in range(x,x+28,1):
        for j in range(y,y+28,1):
            hh[0][k]=dd[i][j]
            k=k+1
    ssd=np.sqrt(error(hh[0],average))
    #dffs=np.sqrt(ssd**2-error(hh[0],(hh[0]-average)*eigenfaces.T*eigenfaces+average))
    dffs=np.sqrt(ssd-error_DFFs(hh[0],(hh[0]-average)*eigenfaces.T))
    return ssd,dffs

ssd = np.zeros((252,252))
dffs = np.zeros((252,252))
for i in range(252):
    for j in range(252):
        ssd[i][j]=compu(i,j)[0]/3482*255
        #dffs[i][j]=compu(i,j)[1]/2482*255
        dffs[i][j]=compu(i,j)[1]/2280*255
        
jj=np.max(ssd)
jjj=np.max(dffs)
print("maxValue",jj,jjj)

print("Show the ssd of the test Image:")
plt.imshow(ssd, cmap='gray')#0=black show best match
plt.show()
plt.close()

print("Show the dffs of the test Image:")
plt.imshow(dffs, cmap='gray')
plt.show()
plt.close()

"""
aa = np.zeros((1,784))
aa[0]=average
res = cv2.matchTemplate(dd, aa.reshape(28,28), cv2.TM_SQDIFF)

plt.imshow(res, cmap='gray')
plt.show()
plt.close()

import pandas as pd
#scores, evals, evecs = PCA(swap_df, 7) = np.dot(eigenfaces.T, df.T).T, eigenvals, eigenfaces
#scores=trainset*eigenfaces
evecs = pd.DataFrame(eigenfaces)
plt.plot(evecs.ix[:, 0:2])
plt.show()
"""