import cv2
import os

import numpy as np

from basicsIndeteails import stack_images

dataPath = 'D:\\open_cv\\data\\morph_Transform'
imgFiles = ['balls.jpg', 'orange.jpg']
imgFile = imgFiles[0]

img = cv2.imread(os.path.join(dataPath, imgFile))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Note: for erosion and dilation transformation, we need to have
# Binary image(either black or white pixels)
_, mask = cv2.threshold(imgGray, 250, 255, cv2.THRESH_BINARY_INV)

# dilation
# the wholes < kernel_size are dilated.
kernel_size = np.ones((5, 5), np.uint8)
imgDilate = cv2.dilate(mask, kernel_size, iterations=6)

# erosion: used to separate object from image.
imgErode = cv2.erode(mask, kernel_size)
# display
stack = stack_images(0.6, [img, imgGray, mask, imgDilate, imgErode])
cv2.imshow('stack', stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
