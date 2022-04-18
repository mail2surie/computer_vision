import cv2
import time
import os
from hand_tracking_module import HandDetector
w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

folder_path = '../../data/finger_images'
my_list = os.listdir(folder_path)
over_lay_list = []
for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    over_lay_list.append(image)

p_time = 0
detector = HandDetector(detection_conf=0.75)
tip_ids = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if lm_list:
        fingers = []

        # Thumb, we can't bend thumb similar to other fingers. Hence, checking if it has moved left or right
        # with the help of 'cx' co-ordinate.
        if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # For other fingers use 'cy' co-ordinate.
        for lm_id in range(1, 5):
            if lm_list[tip_ids[lm_id]][2] < lm_list[tip_ids[lm_id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)

        # overlay the image
        h, w, c = over_lay_list[total_fingers - 1].shape
        img[0:h, 0:w] = over_lay_list[total_fingers - 1]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0),
                      cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
