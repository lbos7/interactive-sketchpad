import cv2
import numpy as np


class Region:
    """Class representing regions of the sketchpad."""

    def __init__(self, pos, size, color, text="", transparency=0.9):
        """
        Initializes a Region object.

        Parameters:
            pos (tuple of ints): The row, col position of the top left corner
                                 of the region.
            size (tuple of ints): The size of the region represented by # of
                                  rows, # of cols.
            color (tuple of ints): The bgr color code for the region.
            text (str): The text to be displayed on the region.
            transparency (float): How transparent the region should be. 0 is
                                  fully transparent and 1 is opaque.
        """

        # Defining class attributes with constructor args
        self.pos = pos
        self.size = size
        self.color = color
        self.text = text
        self.transparency = transparency

    def draw(self, img, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        """
        Draws a region.

        Parameters:
            img (3d numpy array): The input image where the region is drawn.
            text_color (tuple of ints): The bgr color code for the region text.
            font (int): The text font.
            font_scale (float): Scale for text size. 1.0 is normal size and
                                2.0 is double the size.
            thickness (int): Line thickness for text.
        """

        # Indexing rectangle from background image and setting up region rectangle
        bg_rect = img[self.pos[0]:self.pos[0] + self.size[0],
                      self.pos[1]:self.pos[1] + self.size[1]].astype(np.float32)

        reg_rect = np.ones(bg_rect.shape, dtype=np.float32) * np.array(self.color, dtype=np.float32)

        # Creating overlayed rectangle for making regions transparent
        overlayed_rect = cv2.addWeighted(reg_rect,
                                         self.transparency,
                                         bg_rect, 1 - self.transparency,
                                         0,
                                         dtype=cv2.CV_32F)

        # Check to make sure overlayed rectangle is defined
        if overlayed_rect is not None:

            # Replacing background rectangle with overlayed rectangle
            img[self.pos[0]:self.pos[0] + self.size[0],
                self.pos[1]:self.pos[1] + self.size[1]] = overlayed_rect.astype(np.uint8)

            # Finding text size and position based on text, font, font size, and thickness
            text_size = cv2.getTextSize(self.text, font, font_scale, thickness)
            text_pos = (int(self.pos[1] + self.size[1] / 2 - text_size[0][0] / 2),
                        int(self.pos[0] + self.size[0] / 2 + text_size[0][1] / 2))

            # Adding text to image
            cv2.putText(img, self.text, text_pos, font, font_scale, text_color, thickness)

    def contains(self, point):
        """
        Checks if a region contains a point.

        Parameters:
            point (tuple of ints): The point of interest to check.

        Returns:
            (bool): Whether or not the point is in the region
        """
        return (self.pos[0] <= point[0] <= self.pos[0] + self.size[0] and
                self.pos[1] <= point[1] <= self.pos[1] + self.size[1])
