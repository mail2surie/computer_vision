import cv2
from pose_estimation_module import PoseDetector
import os
import numpy as np

# declare
data_path = '../../data/ai_trainer_videos'
file_name = 'bicep_curl.mp4'
cap = cv2.VideoCapture(os.path.join(data_path, file_name))
detector = PoseDetector()
count = 0
direction = 0

while True:
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (620, 480))
        img = detector.find_pose(img)

        lm_list = detector.find_position(img, False)
        if lm_list:
            angle = detector.find_angle(img, 12, 14, 16)  # landmark Id's
            print(angle)

            per = np.interp(angle, [10, 170], [0, 100])
            
            # check for the up down movement.
            if per == 100:
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                if direction == 1:
                    count += 0.5
                    direction = 0
            print(count)

        cv2.putText(img, f'{count}', (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break


