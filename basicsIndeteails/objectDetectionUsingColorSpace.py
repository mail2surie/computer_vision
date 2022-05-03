import cv2
import numpy as np


def nothing(x):
    pass


vid = cv2.VideoCapture(0)
cv2.namedWindow('trackbars')
cv2.createTrackbar('L - H', 'trackbars', 0, 179, nothing)
cv2.createTrackbar('L - S', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('L - V', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('U - H', 'trackbars', 179, 179, nothing)
cv2.createTrackbar('U - S', 'trackbars', 255, 255, nothing)
cv2.createTrackbar('U - V', 'trackbars', 255, 255, nothing)


while True:
    ret, img = vid.read()

    # H - color, S - color concentration, v - Brightness
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos('L - H', 'trackbars')
    l_s = cv2.getTrackbarPos('L - S', 'trackbars')
    l_v = cv2.getTrackbarPos('L - V', 'trackbars')
    u_h = cv2.getTrackbarPos('U - H', 'trackbars')
    u_s = cv2.getTrackbarPos('U - S', 'trackbars')
    u_v = cv2.getTrackbarPos('U - V', 'trackbars')

    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(imgHSV, lower_range, upper_range)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('img', result)
    cv2.imshow('mask', mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()