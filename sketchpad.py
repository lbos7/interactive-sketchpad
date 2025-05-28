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
    sketch_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Empty camera frame.")
            break

        # Flip the frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        hand_tracker.detect_hands(frame)

        for button in buttons:
            button.draw(frame)
            cv2.rectangle(frame,
                          (button.pos[1], button.pos[0]),
                          (button.pos[1] + button.size[1], button.pos[0] + button.size[0]),
                          (255, 255, 255),
                          2)

        # Show the image
        cv2.imshow('Interactive Sketchpad', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
