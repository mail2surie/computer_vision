import cv2
import mediapipe as mp
import time


class FaceMesh:
    def __init__(self, mode=False, max_num_faces=1, refine_landmarks=False, min_det_conf=0.5, min_track_conf=0.5):
        self.mode = mode
        self.max_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf
        self.results = None
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.mode, self.max_faces, self.refine_landmarks,self.min_det_conf,
                                                    self.min_track_conf)
        self.draw_specs = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def find_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_CONTOURS, self.draw_specs,
                                                self.draw_specs)
                face = []
                for lm_id, lm in enumerate(face_lms.landmark):
                    # convert them to pixel, total 468 landmarks
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    # cv2.putText(img, f'{lm_id}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    face.append([lm_id, x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    face_detector = FaceMesh()
    p_time = 0
    while True:
        success, img = cap.read()
        img, faces = face_detector.find_mesh(img)
        if faces:
            print(len(faces))
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break


if __name__ == '__main__':
    main()
