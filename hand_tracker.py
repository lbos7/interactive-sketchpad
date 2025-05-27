import cv2
import mediapipe as mp
import numpy as np


class HandTracker:

    def __init__(self, static_mode=False, max_hands=2,
                 min_detect_conf=0.7, min_track_conf=0.5):

        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_mode,
                                         max_num_hands=self.max_hands,
                                         min_detection_confidence=self.min_detect_conf,
                                         min_tracking_confidence=self.min_track_conf)

        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, img, visible_landmarks=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = self.hands.process(img_rgb)

        if processed_img.multi_hand_landmarks:
            for landmark in processed_img.multi_hand_landmarks:
                if visible_landmarks:
                    self.mp_draw.draw_landmarks(img,
                                                landmark,
                                                self.mp_hands.HAND_CONNECTIONS)

        return img

    # def 
