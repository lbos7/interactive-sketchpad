import cv2
import mediapipe as mp
import numpy as np
from math import dist


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

        self.left_landmarks = []
        self.right_landmarks = []
        self.landmarks = []
        self.pos_list = []

    def detect_hands(self, img, visible_landmarks=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = img_rgb.shape
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            # for i, landmarks in enumerate(self.results.multi_hand_landmarks):
            #     handedness = self.results.multi_handedness[i].classification[0]
            #     label = handedness.label
            #     score = handedness.score

                # if score > self.min_lr_conf:
                #     if label == "Left":
                #         self.left_landmarks = landmarks
                #     elif label == "Right":
                #         self.right_landmarks = landmarks
            self.landmarks = self.results.multi_hand_landmarks[len(self.results.multi_hand_landmarks) - 1]
            if visible_landmarks:
                self.mp_draw.draw_landmarks(img,
                                            self.landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)

        return img

    def get_pos(self):
        # left_pos_list = []
        # right_pos_list = []

        # for landmark in self.left_landmarks:
        #     row, col = int(landmark.y * self.img_h), int(landmark.x * self.img_w)
        #     left_pos_list.append((row, col))

        # for landmark in self.right_landmarks:
        #     row, col = int(landmark.y * self.img_h), int(landmark.x * self.img_w)
        #     right_pos_list.append((row, col))

        # pos_dict = {"Left": left_pos_list, "Right": right_pos_list}
        # return pos_dict
        pos_list = []
        if self.landmarks:
            for landmark in self.landmarks.landmark:
                row, col = int(landmark.y * self.img_h), int(landmark.x * self.img_w)
                pos_list.append((row, col))
        self.pos_list = pos_list
        return pos_list

    def get_extended_fingers(self):

        # pos_dict = self.get_pos()
        # left_pos_list = pos_dict["Left"]
        # right_pos_list = pos_dict["Right"]

        # left_extend_list = []
        # right_extend_list = []

        tip_indices = [4, 8, 12, 16, 20]
        extend_list = []

        # if left_pos_list != []:

        #     for i in tip_indices:

        #         left_extend_list.append(
        #             dist(left_pos_list[i], left_pos_list[0]) > dist(left_pos_list[i - 1], left_pos_list[0])
        #             and dist(left_pos_list[i], left_pos_list[0]) > dist(left_pos_list[i - 2], left_pos_list[0])
        #             and dist(left_pos_list[i], left_pos_list[0]) > dist(left_pos_list[i - 3], left_pos_list[0]))

        # if right_pos_list != []:

            # for i in tip_indices:
            #     right_extend_list.append(
            #         dist(right_pos_list[i], right_pos_list[0]) > dist(right_pos_list[i - 1], right_pos_list[0])
            #         and dist(right_pos_list[i], right_pos_list[0]) > dist(right_pos_list[i - 2], right_pos_list[0])
            #         and dist(right_pos_list[i], right_pos_list[0]) > dist(right_pos_list[i - 3], right_pos_list[0]))

        # extened_dict = {"Left": left_extend_list, "Right": right_extend_list}

        if self.pos_list:
            for i in tip_indices:
                if i == 4:
                    # Calculate angle at joint 2-3-4
                    a1 = np.array(self.pos_list[4]) - np.array(self.pos_list[3])
                    b1 = np.array(self.pos_list[2]) - np.array(self.pos_list[3])
                    cos_angle1 = np.dot(a1, b1) / (np.linalg.norm(a1) * np.linalg.norm(b1) + 1e-6)
                    angle_2_3_4 = np.degrees(np.arccos(np.clip(cos_angle1, -1.0, 1.0)))

                    # Calculate angle at joint 1-2-3
                    a2 = np.array(self.pos_list[3]) - np.array(self.pos_list[2])
                    b2 = np.array(self.pos_list[1]) - np.array(self.pos_list[2])
                    cos_angle2 = np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-6)
                    angle_1_2_3 = np.degrees(np.arccos(np.clip(cos_angle2, -1.0, 1.0)))

                    # Check if both angles indicate the thumb is extended
                    extended = (155 <= angle_2_3_4 <= 180) and (170 <= angle_1_2_3 <= 180)
                    extend_list.append(extended)
                else:
                    extend_list.append(
                        dist(self.pos_list[i], self.pos_list[0]) > dist(self.pos_list[i - 1], self.pos_list[0])
                        and dist(self.pos_list[i], self.pos_list[0]) > dist(self.pos_list[i - 2], self.pos_list[0])
                        and dist(self.pos_list[i], self.pos_list[0]) > dist(self.pos_list[i - 3], self.pos_list[0]))



        return extend_list
