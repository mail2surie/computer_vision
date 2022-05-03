import cv2
import os
import numpy as np
from basicsIndeteails import stack_images

# define variables
# define variables
dataDir = 'D:\\open_cv\\data\\basicDetails'
car = 'car.jpg'
road = 'road.jpg'
# car's outer points.
roi = [(79, 128), (128, 105), (180, 96), (218, 90), (240, 88), (282, 63), (314, 55), (351, 49), (399, 44), (439, 47),
       (461, 54), (494, 70), (522, 96), (544, 113), (561, 128), (566, 153), (571, 184), (570, 216), (561, 230),
       (524, 233), (516, 221), (420, 233), (406, 255), (385, 269), (348, 261), (334, 248), (317, 250), (209, 247),
       (190, 239), (84, 234), (66, 235), (63, 224), (59, 208), (58, 189), (66, 160)]


# read images
car = cv2.imread(os.path.join(dataDir, dataDir, car))
road = cv2.imread(os.path.join(dataDir, dataDir, road))
# create a car mask of polygon shape
carMask = np.zeros_like(car)
npRoi = np.array(roi, np.int32).reshape(-1, 1, 2)
carMask = cv2.fillPoly(carMask, [npRoi], (255, 255, 255))
carResult = cv2.bitwise_and(car, carMask)

# create road mask for car place
roadMask = cv2.bitwise_not(carMask)
roadMask = cv2.bitwise_and(road, roadMask)
# combine both the result.
result = cv2.bitwise_or(roadMask, carResult)
# cv2.polylines(car, [npRoi], True, (255, 0, 0), 2)
# for i in roi:
#    cv2.circle(car, i, 5, (255, 0, 255), 2)
stack = stack_images(0.5, [road, car, carMask, carResult])
stack1 = stack_images(0.5, [roadMask, result])
cv2.imshow('stack', stack)
cv2.imshow('stack1', stack1)
cv2.waitKey(0)
cv2.destroyAllWindows()
