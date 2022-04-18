import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for lm_id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            # print(lm_id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bbox_class = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            # extract bbox and drawing without using mp_draw
            bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.height
                                                                                                      * h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 2)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
