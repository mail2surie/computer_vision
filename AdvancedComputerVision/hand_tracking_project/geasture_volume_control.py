import cv2
import time
import numpy as np
from hand_tracking_module import HandDetector
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# set webcam params
w_cam = 640
h_cam = 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

pre_time = 0
detector = HandDetector(detection_conf=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# print(volume.GetVolumeRange()) # (-65.25, 0.0, 0.03125)
volume_range = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
min_vol = volume_range[0]
max_vol = volume_range[1]
vol, vol_bar, vol_per = 0, 400, 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    # extract landmarks list
    lm_list = detector.find_position(img, draw=False)
    if lm_list:
        x1, y1 = lm_list[4][1], lm_list[4][2]  # cx, cy of thumb
        x2, y2 = lm_list[8][1], lm_list[8][2]  # cx, cy of index
        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
        # line centre
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2,
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # Line length
        line_len = math.hypot(x2 - x1, y2 - y1)
        # print(line_len)

        if line_len < 50:
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Hand_range between 50 - 300
        vol = np.interp(line_len, [50, 300], [min_vol, max_vol])
        vol_bar = np.interp(line_len, [50, 300], [400, 150])
        vol_per = np.interp(line_len, [50, 300], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'FPS:{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1,  (255, 0, 0), 3)

    cur_time = time.time()
    fps = 1 / (cur_time - pre_time)
    pre_time = cur_time

    cv2.putText(img, f'FPS:{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
