import cv2
import mediapipe as mp
import numpy as np


class HandTracker:

    def __init__(self, static_mode=False, max_hands=2,
                 min_detect_conf=0.7, min_track_conf=0.5, min_lr_conf=0.8):

        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf
        self.min_lr_conf = min_lr_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_mode,
                                         max_num_hands=self.max_hands,
                                         min_detection_confidence=self.min_detect_conf,
                                         min_tracking_confidence=self.min_track_conf)

        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, img, visible_landmarks=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = img_rgb.shape
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for i, landmarks in enumerate(self.results.multi_hand_landmarks):
                handedness = self.results.multi_handedness[i].classification[0]
                label = handedness.label
                score = handedness.score

                if score > self.min_lr_conf:
                    if label == "Left":
                        self.left_landmarks = landmarks
                    elif label == "Right":
                        self.right_landmarks = landmarks

                if visible_landmarks:
                    self.mp_draw.draw_landmarks(img,
                                                landmarks,
                                                self.mp_hands.HAND_CONNECTIONS)

        return img

    def get_pos(self):
        left_pos_list = []
        right_pos_list = []

        for landmark in self.left_landmarks:
            row, col = int(landmark.y * self.img_h), int(landmark.x * self.img_w)
            left_pos_list.append((row, col))

        for landmark in self.right_landmarks:
            row, col = int(landmark.y * self.img_h), int(landmark.x * self.img_w)
            right_pos_list.append((row, col))

        pos_dict = {"Left": left_pos_list, "Right": right_pos_list}
        return pos_dict
