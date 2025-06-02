import cv2
import numpy as np
from hand_tracker import HandTracker
from region import Region


def create_buttons(starting_pos=(0, 0), button_size=(100, 100)):
    """
    Creates a list of region objects representing buttons.

    Parameters:
        starting_pos (tuple of ints): The row, col position of the first
                                      button's top left pixel.
        button_size (tuple of ints): The button sizes in # of rows, # of cols.

    Returns:
        (list of regions): List of sketchpad buttons.
    """

    # Indexing starting_pos arg
    starting_row = starting_pos[0]
    starting_col = starting_pos[1]

    # Creating buttons
    button_list = [Region(starting_pos, button_size, (0, 0, 255)),
                   Region((starting_row, starting_col + 1*button_size[1]), button_size, (255, 0, 0)),
                   Region((starting_row, starting_col + 2*button_size[1]), button_size, (0, 255, 0)),
                   Region((starting_row, starting_col + 3*button_size[1]), button_size, (0, 255, 255)),
                   Region((starting_row, starting_col + 4*button_size[1]), button_size, (132, 42, 78)),
                   Region((starting_row, starting_col + 5*button_size[1]), button_size, (255, 255, 255)),
                   Region((starting_row, starting_col + 6*button_size[1]), button_size, (1, 1, 1)),
                   Region((starting_row, starting_col + 7*button_size[1]), button_size, (128, 128, 128), text="Eraser"),
                   Region((starting_row, starting_col + 8*button_size[1]), button_size, (128, 128, 128), text="Clear"),
                   Region((starting_row, 1280 - button_size[1]), button_size, (0, 0, 128), text="Exit")]

    return button_list


def slider_map(pos, lower_col=930, upper_col=1150, min_size=5, max_size=40):
    """
    Adjusts cursor size and slider position based on fingertip position.

    Parameters:
        pos (tuple of ints): The row, col position of a fingertip.
        lower_col (int): The leftmost possible column number.
        upper_col (int): The rightmost possible column number.
        min_size (int): The minimum cursor size.
        max_size (int): The maximum cursor size.

    Returns:
        (int), (int): The cursor size and slider position.
    """

    # Generate a list of cols based on bounds and number of points
    cols = np.linspace(lower_col, upper_col, max_size - min_size + 1)

    # Convert to ints
    cols = cols.astype(int)

    # Find index of column closest to fingertip position
    closest_ind = np.abs(cols - pos[1]).argmin()

    return closest_ind + 5, cols[closest_ind]


def main():
    """
    Main function that launches the interactive sketchpad.

    Captures video from the webcam, tracks hand gestures using MediaPipe,
    and allows drawing on a virtual canvas.
    """

    # Defining HandTracker and Region objects
    hand_tracker = HandTracker()
    buttons = create_buttons()
    sketchpad = Region((100, 0), (720 - 100, 1280), (255, 255, 255), transparency=0.0)
    slider = Region((0, 900), (100, 280), (255, 255, 255), transparency=0.0)

    # Setting up empty image for drawings
    sketch_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Initial cursor size and slider position
    cursor_size = 5
    slider_x = slider.pos[1] + 30

    # Initial conditions for toggle, color, previous landmark positions, thumb state, and exiting
    drawing = False
    current_color_button = buttons[5]
    current_color = current_color_button.color
    prev_pos_list = [(0, 0)] * 4
    prev_thumb_state = False
    exit = False

    # Set up webcam feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Loop for running the sketchpad
    while cap.isOpened():

        # Read a frame from the webcam; end the loop if frame is not returned
        success, frame = cap.read()
        if not success:
            print("Empty camera frame.")
            break

        # Flip the frame horizontally for mirror view
        frame = cv2.flip(frame, 1)

        # Hand detection, position update, and extended finger check
        hand_tracker.detect_hands(frame)
        hand_landmark_pos = hand_tracker.get_pos()
        extended_fingers = hand_tracker.get_extended_fingers()
        extended_ind = np.where(np.array(extended_fingers))[0]

        # Toggle for turning drawing on/off using thumb
        if 0 in extended_ind and not prev_thumb_state:

            # Change drawing state and update previous thumb state
            drawing = not drawing
            prev_thumb_state = True

        elif 0 in extended_ind:

            # Update previous thumb state
            prev_thumb_state = True

        else:

            # Update previous thumb state
            prev_thumb_state = False

        # Case if drawing functionality is activated
        if drawing:

            # Add text to frame if drawing functionality is turned on
            text_size = cv2.getTextSize("Drawing", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_pos = (int(1280 - text_size[0][0]), int(720 - text_size[0][1]))
            cv2.putText(frame, "Drawing", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Loop through extended fingers and draw
            for i in extended_ind:

                # Get fingertip position of extended finger
                pos = hand_landmark_pos[4 * (i + 1)]

                # Check if drawing should occur
                if i != 0 and sketchpad.contains(pos) and sketchpad.contains(prev_pos_list[i - 1]) and (prev_pos_list[i - 1] != (0, 0)):

                    # Draw line between current and previous fingertip positions
                    cv2.line(sketch_img,
                             (prev_pos_list[i - 1][1], prev_pos_list[i - 1][0]),
                             (pos[1], pos[0]),
                             current_color,
                             cursor_size)

        # Check for button presses and update previous landmark positions (only executes if hand landmarks have been detected)
        if hand_landmark_pos:

            # Loop through each fingertip index
            for i in range(8, 20 + 1, 4):

                # Current fingertip position
                pos = hand_landmark_pos[i]

                # Case for index finger
                if i == 8:

                    # Loop through each button in list
                    for j in range(0, 10):

                        # Get current button
                        button = buttons[j]

                        # Check if button is pressed
                        if button.contains(pos) and not button.contains(prev_pos_list[int(i / 4) - 2]):

                            # Clear drawing if clear button is pressed
                            if j == 8:
                                sketch_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                                continue

                            # Set exit variable to true if exit button is pressed
                            if j == 9:
                                exit = True
                                break

                            # Update current color button
                            current_color_button = button

                            # Change color to black if eraser button is pressed
                            if j == 7:
                                current_color = (0, 0, 0)

                            # Update color for all other buttons
                            else:
                                current_color = current_color_button.color

                    # Check if fingertip is in slider region and update if so       
                    if slider.contains(pos):
                        cursor_size, slider_x = slider_map(pos)

                # Exit loop if exit variable has been updated
                if exit:
                    break

                # Update previous fingertip positions
                prev_pos_list[int(i / 4) - 2] = pos

        # Draw buttons and white borders
        for button in buttons:
            button.draw(frame)
            cv2.rectangle(frame,
                          (button.pos[1], button.pos[0]),
                          (button.pos[1] + button.size[1], button.pos[0] + button.size[0]),
                          (255, 255, 255),
                          2)

        # Draw cyan border around button for selected color
        cv2.rectangle(frame,
                      (current_color_button.pos[1], current_color_button.pos[0]),
                      (current_color_button.pos[1] + current_color_button.size[1], current_color_button.pos[0] + current_color_button.size[0]),
                      (255, 255, 0),
                      6)

        # Draw slider region (Is currently transparent but can be adjusted)
        slider.draw(frame)

        # Draw slider bar
        cv2.rectangle(frame,
                      (slider.pos[1] + 30, int(slider.pos[0] + slider.size[0] / 2 - 1)),
                      (slider.pos[1] + slider.size[1] - 30, int(slider.pos[0] + slider.size[0] / 2 + 1)),
                      (128, 128, 128),
                      -1)

        # Draw slider circle (same size as cursor)
        cv2.circle(frame,
                   (slider_x, int(slider.pos[0] + slider.size[0] / 2)),
                   int(cursor_size / 2),
                   (255, 255, 255),
                   -1)

        # Add text to slider region
        slider_text_size = cv2.getTextSize("Cursor Size", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        slider_text_pos = (int(slider.pos[1] + slider.size[1] / 2 - slider_text_size[0][0] / 2),
                    int(slider.pos[0] + slider_text_size[0][1] + 5))
        cv2.putText(frame, "Cursor Size", slider_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert drawings array to grayscale
        sketch_img_gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)

        # Find image inverse
        _, inv_img = cv2.threshold(sketch_img_gray, 0, 255, cv2.THRESH_BINARY_INV)

        # Convert inverse image to bgr
        inv_img = cv2.cvtColor(inv_img, cv2.COLOR_GRAY2BGR)

        # Add drawings to frame
        frame = cv2.bitwise_and(frame, inv_img)
        frame = cv2.bitwise_or(frame, sketch_img)

        # Show the image
        cv2.imshow('Interactive Sketchpad', frame)

        # Case for exiting loop
        if cv2.waitKey(1) == ord('q') or exit:
            break

    # Cleanup for webcam stream
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
