# -*- coding: utf-8 -*-
"""
Based on code from
    https://colab.research.google.com/github/spmallick/learnopencv/blob/master/PyTorch-Segmentation-torchvision/intro-seg.ipynb
"""

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import models
import numpy as np
import torchvision.transforms as T
import cv2

##class 6 only label class 6 to be red
def decode_segmap(image, nc=21):
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (255, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
      idx = image == l
      r[idx] = label_colors[l, 0]
      g[idx] = label_colors[l, 1]
      b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb


def apply_mask(im, im_pred):
    """
    Overlays the predicted class labels onto an image using the alpha channel.
    This function assumes that the background label is the black color.
    This function is provided as an inspiration for the masking function you should write.
    """
    r_channel, g_channel, b_channel = cv2.split(im_pred)
    alpha_channel = 127 * np.ones(b_channel.shape, dtype=b_channel.dtype)
    # Make background pixels fully transparent
    alpha_channel -= 127 * np.all(im_pred == np.array([0, 0, 0]), axis=2).astype(b_channel.dtype)
    im_pred = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
    mask = Image.fromarray(im_pred, mode='RGBA')
    # masked_img = Image.fromarray(im)#array to image
    masked_img=im
    masked_img.paste(mask, box=None, mask=mask)
   # return np.array(masked_img)
    return masked_img
# define the model
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

# load an image
img = Image.open('ZYUAN@TCD.IE.png')
plt.imshow(img); plt.show()

# transform the image
trf = T.Compose([T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])#resize 256*256 [0,1]]
inp = trf(img).unsqueeze(0)

# pass the input through the net
out = fcn(inp)['out']
#print (out.shape)#1x21xhx2(20labels+background)

# calculate labels
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()#the highest label values->hxw(2D) each pixel means the lable
#print (np.unique(om))#[ 0  6  7 15] 4 lables: bus car person
#print (om.shape)
# show segmentation output
rgb = decode_segmap(om)
plt.imshow(rgb); plt.show()

new_img=apply_mask(img,rgb)
new_img.save('zyuan@tcd.ie_predicted.png')
#plt.imshow(new_img);
#plt.show()


#caculate IOU
def encode_segmap(image, nc=21):# change the mask image to be only have the class 6 in red
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  (b, g, r) = cv2.split(image)
  i,j=b.shape
  #print(b.shape,label_colors[0],i,j)
  color=[3]
  
  color=np.array([r[0][0],g[0][0],b[0][0]])
  #print(type(color),type(label_colors[0]))
  for m in range(i):
      for n in range(j):
          color=(r[m][n],g[m][n],b[m][n])
          if (color==label_colors[6]).all():
              r[m][n]=255
              g[m][n]=0
              b[m][n]=0
              #print(color)
          else:
              r[m][n]=g[m][n]=b[m][n]=0
  rgb = np.stack([r, g, b], axis=2)
  return rgb

target=cv2.imread('ZYUAN@TCD.IE_mask.png')

rgb1 = encode_segmap(target)
plt.imshow(rgb1); plt.show()

(r1, g, b) = cv2.split(rgb1)
(r2, g, b) = cv2.split(rgb)
#print(np.unique(r1))
#print(np.unique(r2))

i,j=r1.shape
intersection=0
union=0
for m in range(i):
    for n in range(j):
        if r1[m][n]==255 and r2[m][n]==255:
            intersection=intersection+1
            union=union+1
        elif r1[m][n]==0 and r2[m][n]==0:
            a=1
        else:
            union=union+1
#print(intersection,union)           
iou_score = intersection / union
print('iou',iou_score)