# Sources:
# https://www.delftstack.com/howto/python/opencv-inrange/
# https://www.youtube.com/watch?v=t71sQ6WY7L4
# https://www.youtube.com/watch?v=_aTC-Rc4Io0
# https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981

import cv2
import numpy as np

def get_green_coord(path):
    # Get the current frame
    frame = cv2.imread(path)
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Change the range for green color detection
    lower_green = (36, 25, 25)
    upper_green = (86, 255, 255)

    # Create a mask for the green color
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Count the number of green pixels in the mask an calculate ratio
    green_pixel_count = cv2.countNonZero(green_mask)
    total_pixel_count = frame.shape[0] * frame.shape[1]
    green_pixel_ratio = green_pixel_count / total_pixel_count
    #print('Ratio of green pixels:', green_pixel_ratio)

    # Find the contours in the image
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Check if there is a contour with the largest area
    if largest_contour is not None:
        # Find the center of the contour
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x = 0
        center_y = 0

    # Show the original image and the mask
    # cv2.imshow('Original Image', frame)
    # cv2.imshow('green mask', green_mask)

    return green_pixel_ratio, center_x, center_y

def get_red_coord(path):
    # Get the current frame
    frame = cv2.imread(path)
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    # Create a mask for the red color
    red_mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    # Create a mask for the red color
    red_mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)

    red_mask = red_mask1+red_mask2
    # Count the number of green pixels in the mask an calculate ratio
    green_pixel_count = cv2.countNonZero(red_mask)
    total_pixel_count = frame.shape[0] * frame.shape[1]
    green_pixel_ratio = green_pixel_count / total_pixel_count
    #print('Ratio of green pixels:', green_pixel_ratio)

    # Find the contours in the image
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Check if there is a contour with the largest area
    if largest_contour is not None:
        # Find the center of the contour
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x = 0
        center_y = 0

    # Show the original image and the mask
    # cv2.imshow('Original Image', frame)
    # cv2.imshow('green mask', green_mask)

    return green_pixel_ratio, center_x, center_y