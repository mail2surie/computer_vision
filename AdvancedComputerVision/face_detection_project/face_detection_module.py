import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, detection_conf=0.75, model_selection=0):
        self.detection_conf = detection_conf
        self.model_selection = model_selection
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(detection_conf, model_selection)
        self.results = None

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        b_boxs = []
        if self.results.detections:
            for lm_id, detection in enumerate(self.results.detections):
                # mp_draw.draw_detection(img, detection)
                # print(lm_id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bbox_class = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                # extract bbox and drawing without using mp_draw
                bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.
                                                                                                          height * h)
                b_boxs.append([lm_id, bbox, detection.score])
                if draw:
                    img = self.fancy_draw(img, bbox)
                # cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img,
                                f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                (255, 0, 255),
                                2)
        return img, b_boxs

    def fancy_draw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top left x, y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()

        img, b_boxs = detector.find_faces(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break


if __name__ == '__main__':
    main()
