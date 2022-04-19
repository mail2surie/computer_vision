import math
import random

import cv2
import os
import numpy as np
import cvzone
from hand_tracking_module import HandDetector

# Data path
data_path = '../../data'
data_file = 'donut.png'

# Web cap setting
cap_w, cap_h = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, cap_w)
cap.set(4, cap_h)

detector = HandDetector(detection_conf=0.8, max_hands=1)


class SnakeGame:

    def __init__(self, food_path):
        self.points = []  # All points of the snake
        self.lengths = []  # Distance between each point
        self.current_length = 0  # Total length of the snake
        self.allowed_length = 150  # Total allowed length.
        self.previous_head = 0, 0  # previous head point.
        self.img_food = cv2.imread(food_path, cv2.IMREAD_UNCHANGED)
        self.food_h, self.food_w, _ = self.img_food.shape
        self.food_points = 0, 0
        self.random_food_location()

        self.score = 0
        self.final_score = 0
        self.game_over = False

    def random_food_location(self):
        self.food_points = \
            random.randint(100, 1000), \
            random.randint(100, 600)

    def reset_game(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.previous_head = 0, 0
        self.score = 0
        self.random_food_location()

    def update(self, img_main, current_head):
        if self.game_over:
            cvzone.putTextRect(img_main, "Game Over", (300, 400), scale=7, thickness=5,  offset=20)
            cvzone.putTextRect(img_main, f'your score : {self.final_score}', (300, 550), scale=7, thickness=5,  offset=20)

        else:
            px, py = self.previous_head
            cx, cy = current_head

            self.points.append([cx, cy])
            # Distance between current and previous points.
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            # Update current length
            self.current_length += distance
            # Update previous head as current for next iteration.
            self.previous_head = cx, cy

            # Length reduction
            if self.current_length > self.allowed_length:
                for idx, length in enumerate(self.lengths):
                    self.current_length -= length
                    self.lengths.pop(idx)
                    self.points.pop(idx)
                    if self.current_length < self.allowed_length:
                        break

            # Check if snake ate food.

            rx, ry = self.food_points
            # print('donut points')
            # print('----------')
            # print(rx - self.food_w // 2, cx, rx + self.food_w // 2)
            # print(ry - self.food_h // 2, cy, ry + self.food_h // 2)
            if rx - self.food_w // 2 < cx < rx + self.food_w // 2 and \
                    ry - self.food_h // 2 < cy < ry + self.food_h // 2:
                self.random_food_location()
                self.allowed_length += 50
                self.score += 1
                print(self.score)

            # Draw snake
            if self.points:
                for idx, point in enumerate(self.points):
                    if idx != 0:
                        cv2.line(img_main,
                                 self.points[idx - 1],
                                 self.points[idx],
                                 (0, 0, 255), 20)
                cv2.circle(img_main, self.points[-1], 20, (200, 0, 200),
                           cv2.FILLED)

            # Check for collision.
            # Using polygon test to check collision.
            pts = np.array(self.points[:-2], np.int32)  # [:-2] - we don't consider last two values for polygon test.
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_main, [pts], False, (0, 200, 0), 3)
            # check if the current head points hitting any of the "pts".

            min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)
            # test if the head has hit the tail.
            if -1 <= min_dist <= 1:
                print("hit")
                self.game_over = True
                self.final_score = self.score
                # Reset the game variables.
                self.reset_game()

            # Draw food
            rx, ry = self.food_points
            img_main = cvzone.overlayPNG(img_main, self.img_food, (rx - self.food_w // 2, ry - self.food_h // 2))
            cvzone.putTextRect(img_main, f'Score : {self.score}', (50, 80), scale=3, thickness=3, offset=10)

        return img_main


game = SnakeGame(os.path.join(data_path, data_file))
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list, bbox = detector.find_position(img)
    if lm_list:
        point_index = lm_list[8][1:]
        # print(point_index)
        img = game.update(img, point_index)
        # cv2.circle(img, point_index, 20, (200, 0, 200), cv2.FILLED)

    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.game_over = False
    elif key & 0XFF == ord('q'):
        break
cv2.destroyAllWindows()
