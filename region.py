import cv2
import numpy as np


class Region:

    def __init__(self, pos, size, color, text="", transparency=0.9):
        self.pos = pos
        self.size = size
        self.color = color
        self.text = text
        self.transparency = transparency

    def draw(self, img, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        bg_rect = img[self.pos[0]:self.pos[0] + self.size[0],
                      self.pos[1]:self.pos[1] + self.size[1]].astype(np.float32)
        reg_rect = np.ones(bg_rect.shape, dtype=np.float32) * np.array(self.color, dtype=np.float32)
        overlayed_rect = cv2.addWeighted(reg_rect,
                                         self.transparency,
                                         bg_rect, 1 - self.transparency,
                                         0,
                                         dtype=cv2.CV_32F)
        if overlayed_rect is not None:
            img[self.pos[0]:self.pos[0] + self.size[0],
                self.pos[1]:self.pos[1] + self.size[1]] = overlayed_rect.astype(np.uint8)
            text_size = cv2.getTextSize(self.text, font, font_scale, thickness)
            text_pos = (int(self.pos[1] + self.size[1] / 2 - text_size[0][0] / 2),
                        int(self.pos[0] + self.size[0] / 2 + text_size[0][1] / 2))
            cv2.putText(img, self.text, text_pos, font, font_scale, text_color, thickness)

    def contains(self, point):
        return (self.pos[0] <= point[0] <= self.pos[0] + self.size[0] and
                self.pos[1] <= point[1] <= self.pos[1] + self.size[1])
