import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_specs = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

p_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lms, mp_face_mesh.FACEMESH_CONTOURS, draw_specs, draw_specs)
            for lm_id, lm in enumerate(face_lms.landmark):
                # convert them to pixel, total 468 landmarks
                h, w, c = img.shape
                x, y = int(lm.x*w), int(lm.y*h)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
