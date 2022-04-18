import cv2
import numpy as np


def get_contours(img):
    contours, hierarchy = cv2.findContours(img,
                                           cv2.RETR_EXTERNAL,  # retrieve the outer corners.
                                           cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(img_contours, cnt, -1, (255, 0, 0), 3)
            # calculate curve length
            peri = cv2.arcLength(cnt, True)
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_cor == 3:
                object_type = 'Tri'
            elif obj_cor == 4:
                asp_ratio = w/float(h)
                if 0.95 < asp_ratio < 1.05:
                    object_type = 'Square'
                else:
                    object_type = 'Rectangle'
            elif obj_cor > 4:
                object_type = 'circle'
            else:
                object_type = 'None'
            cv2.rectangle(img_contours, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_contours, object_type,
                        (x+(w//2)-10, y+(h//2)-10),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        0.5, (0, 0, 0), 2)


img = cv2.imread('data/shapes.jpg')

print(img.shape)
img = cv2.resize(img,
                 (int(img.shape[0]*.60),
                  int(img.shape[1]*.40)))
img_contours = img.copy()

# pre process
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_grey,
                            (3, 3),
                            1)
img_canny = cv2.Canny(img_blur,
                      50, 50)
# img_blank = np.zeros_like(img)
get_contours(img_canny)
cv2.imshow('img_original', img)
cv2.imshow('img_grey', img_grey)
cv2.imshow('img_blur', img_blur)
cv2.imshow('img_canny', img_canny)
cv2.imshow('img_canny', img_canny)
cv2.imshow('img_contours', img_contours)
cv2.waitKey(0)
