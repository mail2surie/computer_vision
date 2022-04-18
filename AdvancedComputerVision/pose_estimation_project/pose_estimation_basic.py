import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('../../data/four_min_workout.mp4')
p_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img,
                               results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)
        for lm_id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            #cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)


    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
