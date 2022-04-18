import cv2
import mediapipe as mp
import time
import math


class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True, en_segmentation=False,
                 smooth_segmentation=True, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.en_segmentation = en_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode,
                                      self.complexity,
                                      self.smooth,
                                      self.en_segmentation,
                                      self.smooth_segmentation,
                                      self.detection_conf,
                                      self.tracking_conf)
        self.results = None
        self.lm_lst = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img,
                                            self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lm_lst = []
        if self.results.pose_landmarks:
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_lst.append([lm_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lm_lst

    def find_angle(self, img, p1, p2, p3, draw=True):
        # get the landmarks
        _, x1, y1 = self.lm_lst[p1]
        _, x2, y2 = self.lm_lst[p2]
        _, x3, y3 = self.lm_lst[p3]
        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        # print(angle)
        # draw
        if draw:
            # connecting all three landmarks
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            # highlight the landmarks
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # Write the angle value near middle landmark
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture('../../data/four_min_workout.mp4')
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_lst = detector.find_position(img, False)
        if lm_lst:
            # print(lm_lst[14])  # right elbow
            lm_id, cx, cy = lm_lst[14]
            if lm_id == 14:
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
