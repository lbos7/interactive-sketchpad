# Interactive Sketchpad

The Interactive Sketchpad is an application that allows a user to draw on a virtual canvas using their hand.

## Features

- 7 selectable colors
- Multi-finger drawing
- Erasing
- Clear screen
- Adjustable cursor
- Exit the application without touching the keyboard

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/lbos7/interactive-sketchpad.git
    ```

2. **Install the required dependencies:**
    ```sh
    pip install opencv-python
    pip install mediapipe
    pip install numpy
    ```

## Usage

1. **Run the sketchpad.py script:**
    ```sh
    python sketchpad.py
    ```

2. **Interact with the virtual canvas:**
    - The application will open a window displaying the webcam feed
    - The first hand you raise will be detected as the "drawing hand"
    - Extend your thumb to toggle the cursor on and off
    - Use your index fingertip to change colors, press buttons, or adjust sliders

3. **Use your finger to hit the 'Exit' button on-screen or press 'q' on the keyboard to quit the application.**

## Demo
[Watch a demo](https://www.youtube.com/watch?v=eManR3R7KWY)

