import cv2
import opencv
import numpy as np
from hand_tracking_module import HandDetector
import time
import autopy

# webcam params
cam_w, cam_h = 640, 480
frame_reduction = 100
smoothening = 5
cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

p_time = 0
p_loc_x, p_loc_y = 0, 0
c_loc_x, c_loc_y = 0, 0

detector = HandDetector(max_hands=1)
scr_w, scr_h = autopy.screen.size()
print(scr_w, scr_h) # 1280.0, 720.0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if success:
        img = detector.find_hands(img)
        # 1. Find hand landmarks.
        lm_list, bbox = detector.find_position(img)

        # 2. Get the tip of index and middle finger.
        if lm_list:
            _, x1, y1 = lm_list[8]
            _, x2, y2 = lm_list[12]
            #print(x1, y1, x2, y2)

        # 3. check which fingers are up.
        fingers = detector.fingers_up()
        # print(fingers)
        # 4. Only index finger : Moving mode
        if fingers:
            cv2.rectangle(img, (frame_reduction, frame_reduction),
                          (cam_w - frame_reduction, cam_h - frame_reduction),
                          (255, 0, 255), 2)
            if fingers[1] and not fingers[2]:
                # 5. Convert co-ordinates.

                x3 = np.interp(x1, (frame_reduction, cam_w - frame_reduction), (0, scr_w))
                y3 = np.interp(y1, (frame_reduction, cam_h - frame_reduction), (0, scr_h))

                # 6. Smoothen values.
                c_loc_x = p_loc_x + (x3 - p_loc_x) / smoothening
                c_loc_y = p_loc_y + (y3 - p_loc_y) / smoothening
                # 7. Mov mouse.
                # autopy.mouse.move(scr_w - c_loc_x, c_loc_y)
                autopy.mouse.move(c_loc_x, c_loc_y)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                p_loc_x, p_loc_y = c_loc_x, c_loc_y
            # 8. Both index and middle fingers are up : clicking mode
            if fingers[1] and fingers[2]:
                length, img, line_info = detector.find_distance(8, 12, img)
                print(length)
                if length < 40:
                    cv2.circle(img, (line_info[4], line_info[4]),
                               15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

            # 9. Find distances between fingers.
        # 10. Click mouse if distance is short.
        # 11. frame Rate.
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS:{int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 12. Display
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
