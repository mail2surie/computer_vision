import cv2
import os
import numpy as np
from basicsIndeteails import stack_images

dataPath = 'D:\\open_cv\\data\\smoothing'
imgLst = ['balloons_noisy.png', 'carpet.jpg', 'early_1800.jpg', 'lake.jpg']
imgFile = imgLst[-1]
img = cv2.imread(os.path.join(dataPath, imgFile))

# blur img
imgAvg = cv2.blur(img, (21, 21))
imgGaussian = cv2.GaussianBlur(img, (21, 21), 0)
imgMedian = cv2.medianBlur(img, 5)
imgBilateral = cv2.bilateralFilter(img, 5, 250, 250)  # it keeps edges.


# img stack
stackEarly = stack_images(0.4, [img, imgAvg])
stackEarly1 = stack_images(0.4, [img, imgGaussian])
stackEarly2 = stack_images(0.4, [img, imgMedian])
stackEarly3 = stack_images(0.4, [img, imgBilateral])

# display
cv2.imshow('blur', stackEarly)
cv2.imshow('gaussianBlur', stackEarly1)
cv2.imshow('medianBlur', stackEarly2)
cv2.imshow('bilateral', stackEarly3)

cv2.waitKey(0)
cv2.destroyAllWindows()

