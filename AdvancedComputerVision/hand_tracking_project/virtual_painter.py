import cv2
import numpy as np
import time
import os
from hand_tracking_module import HandDetector

folder_path = '../../data/virtual_paint'
my_list = os.listdir(folder_path)
over_lay_list = []
for img_path in my_list:
    image = cv2.imread(os.path.join(folder_path, img_path))
    image = cv2.resize(image, (1280, 126))
    over_lay_list.append(image)
header = over_lay_list[0]
draw_color = (255, 0, 255)
brush_thickness = 15
erase_thickness = 50
xp, yp = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detection_conf=0.85)
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    # flip in order to align with webcam mirror image.
    img = cv2.flip(img, 1)
    # 2. Find Hand Landmarks
    img = detector.find_hands(img, draw=False)
    lm_list = detector.find_position(img, draw=False)
    if lm_list:
        # tip of index and middle fingers.
        _, x1, y1 = lm_list[8]
        _, x2, y2 = lm_list[12]
        # 3. Check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          draw_color, cv2.FILLED)
            print("Selection mode")
            # checking for the click, (work on header position)
            if y1 < 126:
                if 250 < x1 < 450:
                    header = over_lay_list[0]
                    draw_color = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = over_lay_list[1]
                    draw_color = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = over_lay_list[2]
                    draw_color = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = over_lay_list[3]
                    draw_color = (0, 0, 0)

        # 5. If Drawing mode - Index finger is up
        if fingers[1] and not fingers[2]:
            print('Drawing mode')
            cv2.circle(img, (x1, y1), 15,
                       draw_color, cv2.FILLED)
            if not xp and not yp:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, erase_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, erase_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            xp, yp = x1, y1
    # draw on original image rather than canvas
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    # converting into binary and also do inverse.
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Overlay header image.
    img[0:126, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)
    cv2.imshow('img', img)
    cv2.imshow('canvas', img_canvas)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
