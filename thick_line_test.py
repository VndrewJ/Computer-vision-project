# Testing feature detection of a single image
# Uses OpenCV Feature Detection Tutorial
# https://docs.opencv.org/4.x/d7/d66/tutorial_feature_detection.html

# Packages
import cv2
import numpy as np
import os

# Construct the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = "image_datasets\\Set 3"
image_name = "image0303.jpg"
path = os.path.join(current_dir, image_folder, image_name)

def detect_weld_gaps(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Edge Detection (using Sobel filter)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    edges = cv2.threshold(np.abs(sobelx), 50, 255, cv2.THRESH_BINARY)[1]

    # Step 2: Morphological Operations (dilation)
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert dilated_edges to the appropriate format (if necessary)
    dilated_edges = dilated_edges.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Find the Thickest Line
    max_width = 0
    thickest_contour = None
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w > max_width:
            max_width = w
            thickest_contour = contour

    # Step 5: Draw the Thickest Line
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, [thickest_contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Thickest Vertical Line', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_weld_gaps(path)