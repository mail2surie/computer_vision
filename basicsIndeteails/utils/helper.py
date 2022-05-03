import cv2
import numpy as np


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def find_corner_points(biggest_contours):
    corner_contours = np.zeros((4, 1, 2), np.int32)
    biggest_contours = np.squeeze(biggest_contours)
    sum_c = np.sum(biggest_contours, axis=1)
    diff_c = np.diff(biggest_contours, axis=1)
    corner_contours[0] = biggest_contours[np.argmin(sum_c)]
    corner_contours[1] = biggest_contours[np.argmin(diff_c)]
    corner_contours[2] = biggest_contours[np.argmax(diff_c)]
    corner_contours[3] = biggest_contours[np.argmax(sum_c)]
    return corner_contours


def bird_eye_view(contours, width, height):
    pts1 = np.float32([contours[0], contours[1], contours[2], contours[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix, pts1, pts2


def get_boxes_markings(img):
    bbox = []
    rows = np.vsplit(img, 5)
    for row in rows:
        columns = np.hsplit(row, 5)
        for column in columns:
            bbox.append(column)
    return bbox


def show_answers(img_cont, questions, choices, answers, my_answers):
    sec_h, sec_w = img_cont.shape[0] // choices, img_cont.shape[1] // questions
    for i in range(0, 5):
        if answers[i] == my_answers[i]:
            my_color = (0, 255, 0)
            cx, cy = sec_w * answers[i] + sec_w // 2, sec_h * i + sec_h // 2
        else:
            my_color = (255, 0, 255)
            cx, cy = sec_w * answers[i] + sec_w // 2, sec_h * i + sec_h // 2
            cv2.circle(img_cont, (cx, cy), 30, my_color, 5)
            my_color = (0, 0, 255)
            cx, cy = sec_w * my_answers[i] + sec_w // 2, sec_h * i + sec_h // 2

        cv2.circle(img_cont, (cx, cy), 30, my_color, 5)
    return img_cont


def main():
    pass


if __name__ == '__main__':
    main()
