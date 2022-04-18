import cv2
import numpy as np


def empty(argv):
    pass


# track bars
cv2.namedWindow('TrackBars')
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 57, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 109, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 4, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 209, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 20, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
vid = cv2.VideoCapture(0)

while True:
    ret, img = vid.read()
    # img = cv2.imread('data/lambo.jpg')
    img = cv2.resize(img, (400, 300))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)  # extract car pos
    img_result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Original', img)
    # cv2.imshow('HSV', imgHSV)
    cv2.imshow('Mask', mask)
    cv2.imshow('masked_image', img_result)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()