import cv2
import os
import numpy as np

# define variables
dataDir = 'D:\\open_cv\\data\\OMR_data'
omr = 'omr2.jpg'
roi = []
cap = cv2.VideoCapture(0)


def select_roi(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        roi.append((x, y))


def mouse_callback(window_name):
    cv2.setMouseCallback(window_name, select_roi)


def draw_points(point_list, image, draw_poly=False):
    if point_list:
        for i in roi:
            cv2.circle(image, i, 5, (255, 0, 255), 2)
    if draw_poly:
        if len(point_list) > 1:
            np_roi = np.array(point_list, np.int32)
            np_roi = np_roi.reshape(-1, 1, 2)  # in order to draw polylines this shape needs to be maintained.
            cv2.polylines(image, [np_roi], True, (255, 0, 0), 2)
    return image


def main():
    window_name = 'img'

    while True:
        # img = cv2.imread(os.path.join(dataDir, dataDir, omr))
        success, img = cap.read()
        img = draw_points(roi, img)
        cv2.imshow(window_name, img)
        mouse_callback(window_name)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print(roi)
            break


cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
