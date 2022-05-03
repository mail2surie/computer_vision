import cv2
import os
from basicsIndeteails import  stack_images


def nochange(_argv):
    pass


cv2.namedWindow('ad_thresh')
cv2.createTrackbar('black_size', 'ad_thresh', 3, 255, nochange)
cv2.createTrackbar('constant', 'ad_thresh', 1, 255, nochange)
cv2.setTrackbarMin('black_size', 'ad_thresh', 3)
cv2.setTrackbarMin('constant', 'ad_thresh', 1)
cv2.resizeWindow('ad_thresh', (300, 10))

# define variables
dataPath = 'D:\\open_cv\\data'
imgFile = 'book_page.jpg'

while True:
    img = cv2.imread(os.path.join(dataPath, imgFile))
    img = cv2.resize(img, None, fx=.7, fy=.6)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackSize = cv2.getTrackbarPos('black_size', 'ad_thresh')
    constant = cv2.getTrackbarPos('constant', 'ad_thresh')
    if blackSize % 2 == 0:
        blackSize += 1
    imgGaussian = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blackSize,
                                        constant)
    imgThMean = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blackSize, constant)

    stackRaw = stack_images(0.4, [img, imgGray])
    adaptiveTh = stack_images(0.4, [img, imgGaussian, imgThMean])
    cv2.imshow('stackRaw', stackRaw)
    cv2.imshow('adaptiveTh', adaptiveTh)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

