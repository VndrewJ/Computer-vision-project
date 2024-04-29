# Testing feature detection of a single image
# Uses OpenCV Feature Detection Tutorial
# https://docs.opencv.org/4.x/d7/d66/tutorial_feature_detection.html

# Packages
import cv2 as cv
import numpy as np
import os

# Construct the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = "image_datasets\\Set 3"
image_name = "image0303.jpg"
path = os.path.join(current_dir, image_folder, image_name)

def detect_weld_gaps(image_path):
    # Read image and convert to grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Apply gaussian blur
    blurred = cv.GaussianBlur(image, (5, 5), 0)

    # Use canny edge detection
    edges = cv.Canny(blurred, 30, 150)

    # Create a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological closing to close small gaps
    closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Find contours in the closed edges
    contours, _ = cv.findContours(closed_edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small noise
    min_area = 100  # Adjust as needed
    weld_gap_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    # Filter contours based on aspect ratio to remove non-weld-gap lines
    aspect_ratio_threshold = 10  # Adjust as needed
    filtered_contours = []
    for cnt in weld_gap_contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio < aspect_ratio_threshold:
            filtered_contours.append(cnt)

    # Draw the contours on the original image
    result = image.copy()
    cv.drawContours(result, weld_gap_contours, -1, (0, 255, 0), 2)

    # Display the result
    cv.imshow('Weld Gap Detection', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

detect_weld_gaps(path)