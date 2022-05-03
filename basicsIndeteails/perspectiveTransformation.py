import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

warpPoints = [(243, 257), (411, 257), (245, 447), (424, 446)]
cap = cv2.VideoCapture(0)
try:
    while True:
        success, img = cap.read()
        if success:
            h, w, _ = img.shape
            img = cv2.flip(img, 1)
            pts1 = np.float32(warpPoints)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgPer = cv2.warpPerspective(img, matrix, (w, h))
            # draw wrap points.
            for center in warpPoints:
                cv2.circle(img, center, 5, (255, 0, 255), 2)
            cv2.imshow('img', img)
            cv2.imshow('imgPer', imgPer)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

except:
    print('User cancelled or video failed.')
    cap.release()
    cv2.destroyAllWindows()
