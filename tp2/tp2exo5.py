import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betanbinom
from tensorboard.plugins.histogram.summary import histogram

im1 = cv2.imread('../Images/compteur.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../Images/compteur.jpg')

#
# hist1 = cv2.calcHist([im1], [0], None, [256], [0, 256])
#
#
# hist2_B = cv2.calcHist([im2], [0], None, [256], [0, 256])
# hist2_G = cv2.calcHist([im2], [1], None, [256], [0, 256])
# hist2_R = cv2.calcHist([im2], [2], None, [256], [0, 256])
#
#
# plt.figure(figsize=(12,6))
#
#
# plt.subplot(1, 2, 1)
# plt.title("Histogram of Grayscale Image")
# plt.plot(hist1, color='black')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Number of Pixels')
#
# plt.subplot(1, 2, 2)
# plt.title("Histograms of Colour Image")
# plt.plot(hist2_B, color='blue', label='Blue Channel')
# plt.plot(hist2_G, color='green', label='Green Channel')
# plt.plot(hist2_R, color='red', label='Red Channel')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Number of Pixels')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

#
im3=cv2.cvtColor(im2, cv2.COLOR_BGR2LAB)
# l,a,b=np.split(im3,3)
l=cv2.split(im3)[0]
a=cv2.split(im3)[1]
b=cv2.split(im3)[2]

r=cv2.normalize(l,None,0,300,cv2.NORM_MINMAX)

im4=cv2.merge([r,a,b])
im5=cv2.cvtColor(im4, cv2.COLOR_LAB2BGR)
cv2.imshow('im originale',im2)
cv2.imshow('im4',im5)



def expansionhisto (im, seuil, ymin, ymax):
    if len(im.shape) == 3:
        b, g, r = cv2.split(im)
        b2 = expansionhisto(b, seuil, ymin, ymax)
        g2 = expansionhisto(g, seuil, ymin, ymax)
        r2 = expansionhisto(r, seuil, ymin, ymax)
        return cv2.merge([b2, g2, r2])
    h,w=im.shape[:2]
    histo=np.zeros(256)
    for i in range(h):
        for j in range(w):
            histo[im[i,j]]=histo[im[i,j]]+1

    xmin=0
    xmax=255

    for i in range (256):
      if histo[i]>seuil:
        xmin=i;break;
    alpha=((ymin*xmax)-(ymax*xmin)/(xmax-xmin)) if xmax-xmin>0 else 1
    beta=(ymax-ymin)/(xmax-xmin) if xmax-xmin>0 else 1
    imout=np.zeros_like(im)
    for i in range(h):
      for j in range(w):
          imout[i,j]=alpha + beta* im[i,j]
          if im[i,j]<0 or im[i,j]>255:
              imout[i,j]=im[i,j]

    return imout

r2=expansionhisto(l, 0,0,255)
im7=cv2.merge([r2,a,b])
im8=cv2.cvtColor(im7, cv2.COLOR_LAB2BGR)
cv2.imshow('im8',im8)
cv2.waitKey(0)
cv2.destroyAllWindows()