import cv2
import numpy as np

img = cv2.imread('data/img.png')
# convert to gray scale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_grey, (3, 3), 2)
# Edge detector
img_canny = cv2.Canny(img, 100,200)
# dilation, helps to increase the thickness of edges.
img_dilate = cv2.dilate(img_canny,
                        np.ones((3, 3), np.uint8),
                        iterations=5)
img_eroded = cv2.erode(img_dilate,
                       np.ones((3, 3), np.uint8),
                       iterations=5)
# erosion, opposite of dilation
cv2.imshow('grey_image', img_grey)
cv2.imshow('blur', img_blur)
cv2.imshow('canny_edge', img_canny)
cv2.imshow('dilation', img_dilate)
cv2.imshow('img_eroded', img_eroded)
cv2.waitKey(0)
