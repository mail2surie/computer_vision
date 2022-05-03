import cv2
import os
from basicsIndeteails import stack_images


def no_changes(argv):
    pass


# create trackbar
cv2.namedWindow('trackbar')
cv2.createTrackbar('threshold', 'trackbar', 0, 255, no_changes)
cv2.createTrackbar('blockSize', 'trackbar', 3, 255, no_changes)
cv2.createTrackbar('C', 'trackbar', 0, 50, no_changes)
cv2.setTrackbarMin('blockSize', 'trackbar', 3)
cv2.resizeWindow('trackbar', (400, 10))

# define variables
dataDir = 'D:\\open_cv\\data'
imgName = 'lady.jpg'

while True:
    img = cv2.imread(os.path.join(dataDir, imgName))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = cv2.getTrackbarPos('threshold', 'trackbar')
    block_size = cv2.getTrackbarPos('blockSize', 'trackbar')
    if block_size % 2 == 0:
        block_size += 1
    C = cv2.getTrackbarPos('C', 'trackbar')
    _, imgThB = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY)
    _, imgThBInv = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY_INV)
    _, imgThTrunc = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_TRUNC)
    _, imgThZero = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_TOZERO)
    _, imgThZeroInv = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_TOZERO_INV)
    adaptiveThreshold = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, block_size, C)
    stack = stack_images(0.4, [img, imgGray, imgThB, imgThBInv])
    stackThreshold = stack_images(0.4, [img, imgThTrunc, imgThZero, imgThZeroInv])
    stackAdaptive = stack_images(0.4, [img, imgGray, adaptiveThreshold])
    cv2.imshow('stack', stack)
    cv2.imshow('stackThreshold', stackThreshold)
    cv2.imshow('stackAdaptive', stackAdaptive)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
