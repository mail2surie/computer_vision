import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()  # only uses RGB format
mp_draw = mp.solutions.drawing_utils

pre_time = 0
cur_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)  # lm gives the width and height ratio
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:  # Wrist base
                    cv2.circle(img, (cx, cy), 15,
                               (255, 0, 255),
                               cv2.FILLED)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cur_time = time.time()
    fps = 1/(cur_time-pre_time)
    pre_time = cur_time

    cv2.putText(img, str(int(fps)), (10, 70),3,
                cv2.FONT_HERSHEY_PLAIN,
                (255, 0, 255),
                3)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
