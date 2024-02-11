import cv2
import numpy as np
from detection import detect_lines #Import image processing libraries

# Initialize exponential moving average (EMA) variables inorder to reduce noise and stabilize the centerline
ema_slope = None
ema_intercept = None

def draw_lines(frame, alpha=0.23):
    global ema_slope, ema_intercept  # Use global variables to retain values
    # Initialize lists to hold the classified lines
    group1 = []
    group2 = []
    lines = detect_lines(frame)

    # Dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate mask boundaries
    x_start = int(width * 0.3)
    x_end = int(width * 0.7)
    y_start = int(height * 0.3)
    y_end = int(height * 0.7)

    # Detect lines in the frame
    lines = detect_lines(frame)

    # Classify lines based on the x-coordinate of their midpoint
    #If the midpoint is on the left half of the image, the line is appended to "group1", otherwise it's appended to "group2"
    # #https://www.geeksforgeeks.org/program-find-mid-point-line/
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            midpoint_x = (x1 + x2) / 2
            if midpoint_x < width / 2:
                group1.append(line)
            else:
                group2.append(line)

    # Calculate average points for each group and draw lines
    def draw_lines_and_calculate_average(group, frame, color=(0, 255, 0)):
        if not group:
            return None, None
        sum_start, sum_end = np.array([0, 0]), np.array([0, 0])
        #After iterating through all lines, it calculates the average start and end points by dividing the sum arrays by the number of lines in the group. 
        for line in group:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), color, 6)
            sum_start += np.array([x1, y1])
            sum_end += np.array([x2, y2])
        avg_start = sum_start / len(group)
        avg_end = sum_end / len(group)
        return avg_start, avg_end

    #Draw lines and calculate average start and end points for each group
    avg_start_group1, avg_end_group1 = draw_lines_and_calculate_average(group1, frame)
    avg_start_group2, avg_end_group2 = draw_lines_and_calculate_average(group2, frame)

    # Calculate the average slope for each group
    #https://stackoverflow.com/questions/41462419/python-slope-given-two-points-find-the-slope-answer-works-doesnt-work
    def calculate_average_slope(group):
        slopes = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1 + 0.001)
                slopes.append(slope)
        return np.mean(slopes) if slopes else None

    slope_group1 = calculate_average_slope(group1)
    slope_group2 = calculate_average_slope(group2)

    # https: // www.youtube.com / watch?v = oXlwWbU8l2o
    # https://www.reddit.com/r/algotrading/comments/irj4rm/how_determine_slope_of_moving_average_ma_line/
    #https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # Check if both slopes are available to calculate EMA
    if slope_group1 is not None and slope_group2 is not None:
        average_slope = (slope_group1 + slope_group2) / 2

        # Calculate Exponential Moving Average (EMA) for slope
        if ema_slope is None:  # Initialize EMA if it's the first frame
            ema_slope = average_slope
        else:  # Update EMA for slope
            ema_slope = alpha * average_slope + (1 - alpha) * ema_slope

        # Calculate the central midpoint and y-intercept for the EMA-smoothed central line
        if avg_start_group1 is not None and avg_start_group2 is not None:
            central_midpoint = ((avg_start_group1 + avg_end_group1) / 2 + (avg_start_group2 + avg_end_group2) / 2) / 2
            y_intercept = central_midpoint[1] - ema_slope * central_midpoint[0]

            # Calculate Exponential Moving Average (EMA) for y-intercept
            if ema_intercept is None:  # Initialize EMA if it's the first frame
                ema_intercept = y_intercept
            else:  # Update EMA for intercept
                ema_intercept = alpha * y_intercept + (1 - alpha) * ema_intercept

            # Calculate points for the EMA-smoothed central line
            y1_ema = ema_slope * x_start + ema_intercept
            y2_ema = ema_slope * x_end + ema_intercept

            # Draw the EMA-smoothed central line
            cv2.line(frame, (x_start, int(y1_ema)), (x_end, int(y2_ema)), (0, 0, 255), 3)

        # Draw the boundary of the mask with thin blue lines
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    return frame
