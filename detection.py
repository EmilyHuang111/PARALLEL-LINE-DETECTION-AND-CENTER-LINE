import cv2
import numpy as np #Import image processing libraries

def detect_lines(frame):
    # Dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate mask boundaries
    # https://stackoverflow.com/questions/61383095/create-mask-or-boundary-from-each-other-in-python3
    x_start = int(width * 0.3)
    x_end = int(width * 0.7)
    y_start = int(height * 0.3)
    y_end = int(height * 0.7)

    # Create a mask of zeros
    #https://numpy.org/doc/stable/reference/generated/numpy.ma.make_mask.html
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Define the ROI (Region of Interest) on the mask
    #https://stackoverflow.com/questions/15424852/region-of-interest-opencv-python
    mask[y_start:y_end, x_start:x_end] = 255

    # Convert to grayscale
    #https://www.quora.com/How-do-you-convert-an-RGB-image-to-grayscale-with-OpenCV-and-or-Python
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the mask to the grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply GaussianBlur to reduce noise and improve edge detection
    #https://hackernoon.com/how-to-implement-gaussian-blur-zw28312m
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)

    # Perform edge detection using Canny
    #https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Use HoughLinesP to detect lines within the masked area
    #https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=200)
    return lines
