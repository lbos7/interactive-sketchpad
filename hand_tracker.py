import cv2
import mediapipe as mp
import numpy as np
from math import dist


class HandTracker:
    """Class for tracking hand movements and recognizing gestures."""

    def __init__(self, static_mode=False, max_hands=2,
                 min_detect_conf=0.7, min_track_conf=0.5):
        """
        Initializes a HandTracker object.

        Parameters:
            static_mode (bool): Whether or not to use MediaPipe's hand model
                                in static mode.
            max_hands (int): The max number of hands that can be identified
                             at a time.
            min_detect_conf (float): Minimum confidence required to detect a
                                     hand.
            min_track_conf (float): Minimum confidence required to track a
                                    hand.
        """

        # Defining class attributes with constructor args
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        # Setting up MediaPipe hand model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_mode,
                                         max_num_hands=self.max_hands,
                                         min_detection_confidence=self.min_detect_conf,
                                         min_tracking_confidence=self.min_track_conf)

        # Adding MediaPipe's drawing utilities for drawing hand landmarks
        self.mp_draw = mp.solutions.drawing_utils

        # Empty lists for keeping track of hand landmarks and positions
        self.landmarks = []
        self.pos_list = []

    def detect_hands(self, img, visible_landmarks=True):
        """
        Detects hands in a given image.

        Parameters:
            img (3d numpy array): The image being checked.
            visible_landmarks (bool): Whether or not to draw landmarks on
                                      detected hands.

        Returns:
            (3d numpy array): The input image (with landmarks drawn if true).
        """

        # Converting image from bgr to rgb and determining rows and columns
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = img_rgb.shape

        # Applying hand model to rgb image
        self.results = self.hands.process(img_rgb)

        # Case for if hands are detected
        if self.results.multi_hand_landmarks:

            # Update landmarks (only for first hand that was detected)
            self.landmarks = self.results.multi_hand_landmarks[len(self.results.multi_hand_landmarks) - 1]

            # Case for drawing landmarks
            if visible_landmarks:
                self.mp_draw.draw_landmarks(img,
                                            self.landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)

        return img

    def get_pos(self):
        """
        Finds the row, col positions of all landmarks.

        Returns:
            (list of tuples): A list tuples representing the row, col position
                              of each hand landmark.
        """

        # Defining empty list
        pos_list = []

        # Case for updating positions (only executes if self.landmarks is not empty)
        if self.landmarks:

            # Looping through each landmark
            for landmark in self.landmarks.landmark:

                # Converting landmark x, y coordiates to row, col and adding to position list
                row, col = int(landmark.y * self.img_h), int(landmark.x * self.img_w)
                pos_list.append((row, col))

        # Updating position list
        self.pos_list = pos_list

        return pos_list

    def get_extended_fingers(self):
        """
        Checks if fingers are extended.

        Returns:
            (list of bools): A list boolean values representing whether or not
                             each finger is extended.
        """

        # Defining tip indices and empty list for bools
        tip_indices = [4, 8, 12, 16, 20]
        extend_list = []

        # Case for checking if fingers are extended (only executes if self.pos_list is not empty)
        if self.pos_list:

            # Looping through each tip index
            for i in tip_indices:

                # Case for thumb
                if i == 4:

                    # Calculate angles at joints
                    a1 = np.array(self.pos_list[4]) - np.array(self.pos_list[3])
                    b1 = np.array(self.pos_list[2]) - np.array(self.pos_list[3])
                    cos_angle1 = np.dot(a1, b1) / (np.linalg.norm(a1) * np.linalg.norm(b1) + 1e-6)
                    angle_2_3_4 = np.degrees(np.arccos(np.clip(cos_angle1, -1.0, 1.0)))
                    a2 = np.array(self.pos_list[3]) - np.array(self.pos_list[2])
                    b2 = np.array(self.pos_list[1]) - np.array(self.pos_list[2])
                    cos_angle2 = np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-6)
                    angle_1_2_3 = np.degrees(np.arccos(np.clip(cos_angle2, -1.0, 1.0)))

                    # Calculate normalized distance to landmark 17
                    thumb_tip_dist = dist(self.pos_list[4], self.pos_list[17])
                    hand_base_dist = dist(self.pos_list[0], self.pos_list[5])

                    # Check if thumb is extended
                    thumb_extended = (
                        150 <= angle_2_3_4 <= 195 and
                        150 <= angle_1_2_3 <= 195 and
                        thumb_tip_dist > 1.2 * hand_base_dist
                    )

                    # Add bool to list
                    extend_list.append(thumb_extended)

                # Case for all other fingers
                else:

                    # Check if tip position is further away from wrist than all other landmarks on finger
                    extend_list.append(
                        dist(self.pos_list[i], self.pos_list[0]) > dist(self.pos_list[i - 1], self.pos_list[0])
                        and dist(self.pos_list[i], self.pos_list[0]) > dist(self.pos_list[i - 2], self.pos_list[0])
                        and dist(self.pos_list[i], self.pos_list[0]) > dist(self.pos_list[i - 3], self.pos_list[0]))

        return extend_list
