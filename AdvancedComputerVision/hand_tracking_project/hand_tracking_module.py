import cv2
import mediapipe as mp
import time
import sys
import math
sys.path.insert(0, '../hand_tracking_project')


class HandDetector:
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_hands,
                                         self.complexity,
                                         self.detection_conf,
                                         self.track_conf)  # only uses RGB format
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        x_list = []
        y_list = []
        bbox = []
        x_min, y_min, x_max, y_max = None, None, None, None
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # if x_list and y_list:
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max
        if draw and bbox:
            cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                          (0, 255, 0), 2)
        return self.lm_list, bbox

    def fingers_up(self):
        fingers = []
        if self.lm_list:
            # Thumb, we can't bend thumb similar to other fingers. Hence, checking if it has moved left or right
            # with the help of 'cx' co-ordinate.
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # For other fingers use 'cy' co-ordinate.
            for lm_id in range(1, 5):
                if self.lm_list[self.tip_ids[lm_id]][2] < self.lm_list[self.tip_ids[lm_id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pre_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if lm_list:
            print(lm_list[4])
        cur_time = time.time()
        fps = 1 / (cur_time - pre_time)
        pre_time = cur_time

        cv2.putText(img, str(int(fps)), (10, 70), 3,
                    cv2.FONT_HERSHEY_PLAIN,
                    (255, 0, 255),
                    3)
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break


if __name__ == '__main__':
    main()
