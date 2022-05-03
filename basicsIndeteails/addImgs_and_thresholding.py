# import modules
import cv2
import os
import numpy as np
from basicsIndeteails import stack_images

# Note: to add two images, they must be of same size.
# define variables
dataDir = 'D:\\open_cv\\data\\basicDetails'
car = 'car.jpg'
road = 'road.jpg'
# create image


# load img
car = cv2.imread(os.path.join(dataDir, car))
road = cv2.imread(os.path.join(dataDir, road))
# imgSum = cv2.add(imgRoad, imgRoad)  # pixel by pixel add.
# weighted image add
# 100% weight is given to first image.
# 20% weight is given to 2nd image.
# imgWeight = cv2.addWeighted(imgRoad, 0.7, imgCar, 0.8, 0)
# in order to add only the car, need to remove the background.
carGray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(carGray, 240, 255, cv2.THRESH_BINARY_INV)
# try with adaptive thresholding
mask = cv2.adaptiveThreshold(carGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
mask_inv = cv2.bitwise_not(mask)

imgSumThreshold = cv2.add(car, car, mask=mask)

# take the road background.
roadMask = cv2.bitwise_and(road, road, mask=mask_inv)
carMask = cv2.bitwise_and(car, car, mask=mask_inv)
result = cv2.bitwise_or(imgSumThreshold, roadMask)

# display
stack = stack_images(0.4, [car, carGray, mask, mask_inv])
stack1 = stack_images(0.4, [imgSumThreshold, roadMask])
stack2 = stack_images(0.4, [car, result])
cv2.imshow('stack', stack)
#cv2.imshow('stack1', stack1)
#cv2.imshow('stack2', stack2)
cv2.waitKey(0)
cv2.destroyAllWindows()
