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
        hor_con = [image_blank]*rows
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


def rect_contours(contours):
    rect_cont = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 20:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            # print('Corner points', len(approx))

            # check for rectangle contours.
            if len(approx) == 4:
                rect_cont.append(i)

    # sort contours based on area
    rect_cont = sorted(rect_cont, key=cv2.contourArea, reverse=True)
    return rect_cont


def get_contours_points(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02*peri, True)
    return approx

def re_order(my_points):
    # re order to find location of corners.
    # origin has small sum
    # [[[106  74]]
    # [[ 79 252]]
    # [[275 277]]
    # [[271 100]]] (4, 1, 2) --> (4, 2)
    # remove redundant middle shape
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)
    add = my_points.sum(1)
    # print(my_points)
    # print(add)
    my_points_new[0] = my_points[np.argmin(add)]  # origin [0, 0]
    my_points_new[3] = my_points[np.argmax(add)]  # [w, h]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]  # [w, 0] -->-ve val
    my_points_new[2] = my_points[np.argmax(diff)]  # [o, h] -->+ve val
    # print(diff)
    return my_points_new


def split_boxes(img):
    boxes = []
    rows = np.vsplit(img, 5)
    for row in rows:
        columns = np.hsplit(row, 5)
        for box in columns:
            boxes.append(box)
    return boxes

def show_answers(img, my_index, grading, answers, questions, choices):
    sec_w = int(img.shape[1] / questions)
    sec_h = int(img.shape[0] / choices)
    for x in range(0, questions):
        my_ans = my_index[x]
        # get the centre position of box.
        cx = (my_ans * sec_w) + sec_w // 2
        cy = (x * sec_h) + sec_h // 2

        # mark right and wrong answers in different colors.
        if grading[x] == 1:
            my_color = (0, 255, 0)
        else:
            # mark the correct answers.
            correct_ans = answers[x]
            cv2.circle(img, ((correct_ans * sec_w) + sec_w // 2,
                             (x * sec_h) + sec_h // 2),
                       15,
                       (0, 255, 0),
                       cv2.FILLED)
            my_color = (0, 0, 255)
        # red color for wrong ans.
        cv2.circle(img, (cx, cy), 30, my_color, cv2.FILLED)
    return img



def main():
    pass


if __name__ == '__main__':
    main()
