# Testing feature detection of a single image
# Uses OpenCV Feature Detection Tutorial
# https://docs.opencv.org/4.x/d7/d66/tutorial_feature_detection.html
# This sucked ^ its outdated

# Packages
import cv2 as cv
import numpy as np

path = 'C:\\Users\\Andrew\\OneDrive - The University of Auckland\\Desktop\\Uni Coursebooks\\Year 4\\709\\Computer-vision-project\\image_datasets\\Set 1\\image0001.jpg'
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

#code works kinda, it does indeed find a very smol feature