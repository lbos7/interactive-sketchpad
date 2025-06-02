import cv2
import numpy as np
from hand_tracker import HandTracker
from region import Region

def create_buttons(starting_pos=(0, 0), button_size=(100, 100)):
    starting_row = starting_pos[0]
    starting_col = starting_pos[1]
    button_list = [Region(starting_pos, button_size, (0, 0, 255)),
                   Region((starting_row, starting_col + 1*button_size[1]), button_size, (255, 0, 0)),
                   Region((starting_row, starting_col + 2*button_size[1]), button_size, (0, 255, 0)),
                   Region((starting_row, starting_col + 3*button_size[1]), button_size, (0, 255, 255)),
                   Region((starting_row, starting_col + 4*button_size[1]), button_size, (132, 42, 78)),
                   Region((starting_row, starting_col + 5*button_size[1]), button_size, (255, 255, 255)),
                   Region((starting_row, starting_col + 6*button_size[1]), button_size, (0, 0, 0)),
                   Region((starting_row, starting_col + 7*button_size[1]), button_size, (128, 128, 128), text="Eraser"),
                   Region((starting_row, starting_col + 8*button_size[1]), button_size, (128, 128, 128), text="Clear")]
    return button_list


def main():

    hand_tracker = HandTracker()
    buttons = create_buttons()
    sketchpad = Region((100, 0), (1080 - 100, 1920), (255, 255, 255), transparency=0.0)
    sketch_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    cursor_size = 5

    drawing = False
    current_color_button = buttons[5]
    current_color = current_color_button.color
    prev_pos_list = [(0, 0)] * 4
    prev_thumb_state = False

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Empty camera frame.")
            break

        # Flip the frame horizontally for selfie view
        frame = cv2.flip(frame, 1)

        hand_tracker.detect_hands(frame)
        hand_landmark_pos = hand_tracker.get_pos()
        extended_fingers = hand_tracker.get_extended_fingers()
        extended_ind = np.where(np.array(extended_fingers))[0]

        # Toggle for turning drawing on/off
        if 0 in extended_ind and not prev_thumb_state:
            drawing = not drawing
            prev_thumb_state = True
        elif 0 in extended_ind:
            prev_thumb_state = True
        else:
            prev_thumb_state = False

        print(extended_fingers)
        print("Drawing State: " + str(drawing))
        print("Prev Thumb State: " + str(prev_thumb_state))




        if drawing:

            for i in extended_ind:
                pos = hand_landmark_pos[4 * (i + 1)]
                if i != 0 and sketchpad.contains(pos) and (prev_pos_list[i - 1] != (0, 0)):
                    cv2.line(sketch_img,
                             (prev_pos_list[i - 1][1], prev_pos_list[i - 1][0]),
                             (pos[1], pos[0]),
                             current_color,
                             cursor_size)

        if hand_landmark_pos:
            for i in range(8, 20 + 1, 4):
                pos = hand_landmark_pos[i]
                if sketchpad.contains(pos):
                    prev_pos_list[int(i / 4) - 2] = pos

        for button in buttons:
            button.draw(frame)
            cv2.rectangle(frame,
                          (button.pos[1], button.pos[0]),
                          (button.pos[1] + button.size[1], button.pos[0] + button.size[0]),
                          (255, 255, 255),
                          2)
            
        cv2.rectangle(frame,
                      (current_color_button.pos[1], current_color_button.pos[0]),
                      (current_color_button.pos[1] + current_color_button.size[1], current_color_button.pos[0] + current_color_button.size[0]),
                      (255, 0, 255),
                      4)

        sketch_img_gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)
        _, inv_img = cv2.threshold(sketch_img_gray, 20, 255, cv2.THRESH_BINARY_INV)
        inv_img = cv2.cvtColor(inv_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv_img)
        frame = cv2.bitwise_or(frame, sketch_img)

        # Show the image
        cv2.imshow('Interactive Sketchpad', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
