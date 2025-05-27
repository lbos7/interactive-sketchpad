import cv2
from hand_tracker import HandTracker

hand_tracker = HandTracker()

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for selfie view
    frame = cv2.flip(frame, 1)

    frame = hand_tracker.detect_hands(frame)

    # Show the image
    cv2.imshow('Hand Detection', frame)

    # Break the loop when ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
