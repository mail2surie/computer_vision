import cv2
import os
import numpy as np
from basicsIndeteails import stack_images

# define variables
# define variables
dataDir = 'D:\\open_cv\\data'
lady = 'lady.jpg'
logo = 'logo.png'
logo_w, logo_h = 200, 100


imgLady = cv2.imread(os.path.join(dataDir, lady))
imgLogo = cv2.imread(os.path.join(dataDir, logo))
# resize the logo to fit on img.
imgLogo = cv2.resize(imgLogo, (logo_w, logo_h))
# create an empty image to add both the images.
ldy_h, ldy_w, ldy_channels = imgLady.shape
imgEmpty = np.zeros((ldy_h, ldy_w, ldy_channels), np.uint8)  # default np.zeros_like creates array in np.uint8 format.
imgEmpty[imgEmpty.shape[0] - logo_h:, imgEmpty.shape[1] - logo_w:] = imgLogo
# blend two images with added weight.
result = cv2.addWeighted(imgLady, 1, imgEmpty, 0.3, 0)
# display
stack = stack_images(0.4, [imgLady, imgLogo, imgEmpty, result])
cv2.imshow('stack', stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
