import cv2
import numpy as np
from drawLines import draw_lines  # Assuming draw_lines is a function you've defined elsewhere

# Start video capture from the webcam
##https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
cap = cv2.VideoCapture(0)  # 1 here denotes the index of the webcam, change it if your webcam has a different index

while True:
    # Read the frame
    ret, frame = cap.read()  # ret is a boolean indicating whether reading was successful
    if not ret:  # If ret is False, it means there's no frame to read, so break the loop
        break

    # Detect lines in the frame using a function called draw_lines
    frame_with_lines = draw_lines(frame)  # Assuming draw_lines() takes a frame and returns a frame with lines drawn on it

    # Display the frame with lines drawn
    #https://stackoverflow.com/questions/41067007/trouble-with-cv2-imshow-function
    cv2.imshow('Webcam Line Detection with Mask', frame_with_lines)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # This waits for a key event for 1 millisecond, if it's 'q', break the loop
        break

# Release the VideoCapture object and close windows
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows

