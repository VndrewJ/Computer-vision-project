# Testing feature detection of a single image
# Uses OpenCV Feature Detection Tutorial
# https://docs.opencv.org/4.x/d7/d66/tutorial_feature_detection.html

# Packages
import cv2 as cv
import numpy as np
import os

# Construct the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = "image_datasets\\Set 1"
image_name = "image0003.jpg"
path = os.path.join(current_dir, image_folder, image_name)

img = cv.imread(path, cv.IMREAD_GRAYSCALE)
if img is None:
    print("bruh theres no image")
    exit(0)

#-- Step 1: Detect the keypoints using ORB Detector
# Initialize ORB
orb = cv.ORB_create()

# Find the keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)

#-- Draw keypoints
img_keypoints = cv.drawKeypoints(img, keypoints, None)

#-- Show detected (drawn) keypoints
cv.imshow('ORB Keypoints', img_keypoints)

cv.waitKey(0)
cv.destroyAllWindows()
