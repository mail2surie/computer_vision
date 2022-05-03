import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

dataPath = 'D:\\open_cv\\data\\histograms'
imgFile = 'sea.jpg'
img = cv2.imread(os.path.join(dataPath, imgFile))
b, g, r = cv2.split(img)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('imgGray', imgGray)
#plt.hist(img.ravel(), 256, [0, 256])
plt.hist(b.ravel(), 256, [0, 256])
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(r.ravel(), 256, [0, 256])
plt.show()




