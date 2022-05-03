import cv2
import os
import numpy as np

# define variables
dataDir = 'D:\\open_cv\\data'
imgFile = 'grid.jpg'

img = cv2.imread(os.path.join(dataDir, imgFile))
h, w, _ = img.shape
cv2.circle(img, (83, 90), 5, (0, 0, 255), -1)
cv2.circle(img, (447, 90), 5, (0, 0, 255), -1)
cv2.circle(img, (83, 472), 5, (0, 0, 255), -1)

pts1 = np.float32([[83, 90], [447, 90], [83, 472]])
pts2 = np.float32([[0, 0], [447, 90], [150, 472]])
matrix = cv2.getAffineTransform(pts1, pts2)
imgWarpAffine = cv2.warpAffine(img, matrix, (w, h))
cv2.imshow('img', img)
cv2.imshow('imgWarpAffine', imgWarpAffine)
cv2.waitKey(0)
cv2.destroyAllWindows()
