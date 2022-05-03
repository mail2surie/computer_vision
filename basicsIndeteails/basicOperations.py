# import modules
import cv2
import os
import numpy as np

# define variables
dataDir = 'D:\\open_cv\\data\\basicDetails'
dataFile = 'flag.png'

# create image


# load img
img = cv2.imread(os.path.join(dataDir, dataFile))
img[:, 90:180, :] = (255, 0, 255)

print(img.shape)

# display
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

