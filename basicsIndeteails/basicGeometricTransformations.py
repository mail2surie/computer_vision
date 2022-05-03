import cv2
import numpy as np
import os


dataPath = 'D:\\open_cv\\data\\basicDetails'
imgFile = 'panda.jpg'


img = cv2.imread(os.path.join(dataPath, imgFile))
h, w, c = img.shape
# image scaling
imgScaled = cv2.resize(img, None, fx=1/2, fy=1/2)
# image translation
matrix = np.float32([[1, 0, 100], [0, 1, 100]])
imgTrans = cv2.warpAffine(img, matrix, (w, h))
# image rotation
matrix_rotate = cv2.getRotationMatrix2D((w/2, h/2), 45, 1.5)
imgRotate = cv2.warpAffine(img, matrix_rotate, (w, h))
cv2.imshow('img', img)
cv2.imshow('imgScaled', imgScaled)
cv2.imshow('imgTrans', imgTrans)
cv2.imshow('imgRotate', imgRotate)
cv2.waitKey(0)
cv2.destroyAllWindows()
