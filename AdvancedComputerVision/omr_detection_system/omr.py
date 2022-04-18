import cv2
import os

import numpy as np
import utils


img_h = 480
img_w = 320
data_folder = '../../data/omr_data'
file_name = 'omr2.jpg'
questions = 5
choices = 5
answers = [1, 2, 0, 1, 4]

# Pre-processing
img = cv2.imread(os.path.join(data_folder, file_name))
img = cv2.resize(img, (img_w, img_h))
img_contours = img.copy()
img_final = img.copy()
img_bigger_contour = img.copy()
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_grey, (3, 3), 1)
img_canny = cv2.Canny(img_blur, 10, 50)

# Finding all contours
contours, hierarchy = cv2.findContours(img_canny,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# Find rectangles
rect_cont = utils.rect_contours(contours)
# corner points of bigger rectangle
biggest_cont = utils.get_contours_points(rect_cont[0])
grade_points = utils.get_contours_points(rect_cont[1])
# print(biggest_cont.shape)


if biggest_cont.size != 0 and grade_points.size != 0:
    cv2.drawContours(img_bigger_contour, biggest_cont,
                     -1, (0, 255, 0), 20)
    cv2.drawContours(img_bigger_contour, grade_points,
                     -1, (255, 0, 0), 20)

    biggest_cont = utils.re_order(biggest_cont)
    grade_points = utils.re_order(grade_points)
    # print(biggest_cont)

    # get the bird eye view using warp
    # bigger_cont
    pts1 = np.float32(biggest_cont)
    pts2 = np.float32([[0, 0], [img_w, 0], [0, img_h], [img_w, img_h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp_colored = cv2.warpPerspective(img, matrix, (img_w, img_h))

    # grade_cont
    ptg1 = np.float32(grade_points)
    ptg2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrix_g = cv2.getPerspectiveTransform(ptg1, ptg2)
    img_grade_warp_colored = cv2.warpPerspective(img, matrix_g, (325, 150))
    # cv2.imshow('img_grade_warp_colored', img_grade_warp_colored)

    # apply threshold to get more pixels on marked answer cells.
    img_warp_colored_grey = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
    img_threshold = cv2.threshold(img_warp_colored_grey, 180, 255, cv2.THRESH_BINARY_INV)[1]

    # Extract the boxes
    boxes = utils.split_boxes(img_threshold)
    #cv2.imshow('box', boxes[2])

    # get the non-zero pixels value.
    my_pixels_value = np.zeros((questions, choices))
    count_c = 0
    count_r = 0
    for image in boxes:
        total_pixels = cv2.countNonZero(image)
        my_pixels_value[count_r][count_c] = total_pixels
        count_c += 1
        if count_c == choices:
            count_r += 1
            count_c = 0
    print(my_pixels_value)

    # finding index values of marking.
    my_index = []
    for x in range(0, questions):
        arr = my_pixels_value[x]
        my_index_val = np.where(arr == np.amax(arr))
        my_index.append(my_index_val[0][0])  # since 1 max element in the array.
    #print(my_index)

    # Grading.
    grading = []
    for x in range(0, questions):
        if answers[x] == my_index[x]:
            grading.append(1)
        else:
            grading.append(0)
    print(grading)

    # final score
    score = (sum(grading) / questions) * 100
    print(score)

    # Displaying Answers.
    img_result = img_warp_colored.copy()
    img_result = utils.show_answers(img_result, my_index, grading, answers, questions, choices)

    # combine the result with original img.
    # markings
    img_raw_drawing = np.zeros_like(img_warp_colored)
    img_raw_drawing = utils.show_answers(img_raw_drawing, my_index, grading, answers, questions, choices)
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    img_inv_warp = cv2.warpPerspective(img_raw_drawing, inv_matrix, (img_w, img_h))

    # grade
    img_raw_grade = np.zeros_like(img_grade_warp_colored)
    cv2.putText(img_raw_grade, str(int(score))+"%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    inv_matrix_g = cv2.getPerspectiveTransform(ptg2, ptg1)
    img_inv_grade_display = cv2.warpPerspective(img_raw_grade, inv_matrix_g, (img_w, img_h))

    img_final = cv2.addWeighted(img_final, 1, img_inv_warp, 1, 0)
    img_final = cv2.addWeighted(img_final, 1, img_inv_grade_display, 1, 0)
    cv2.imshow('grade', img_final)

img_blank = np.zeros_like(img)
img_stack = utils.stack_images(0.5,
                         [img,
                          img_grey,
                          img_blur,
                          img_canny,
                          img_contours,
                          ])
img_stack1 = utils.stack_images(0.5,
                         [img_bigger_contour,
                          img_threshold,
                          img_result,
                          img_raw_drawing,
                          img_inv_warp])

img_stack2 = utils.stack_images(0.5,
                                [img,
                                 img_final])


# cv2.imshow('Img_stack', img_stack)
# cv2.imshow('Img_stack1', img_stack1)
cv2.imshow('Img_stack2', img_stack2)


cv2.waitKey(0)
cv2.destroyAllWindows()
