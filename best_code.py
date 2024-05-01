# %%

import cv2
import os
import numpy as np

# Get the current working directory
path = os.getcwd()

# Define input and output directories
inputPar = os.path.join(path, 'image_datasets/Set 3/')
outPar = os.path.join(path, 'output_images/')

os.makedirs(outPar, exist_ok=True)

# List all files in the input directory
files = os.listdir(inputPar)

files = [files[0]]

for file in files:
    fitem = os.path.join(inputPar, file)
    img = cv2.imread(fitem)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kernal used to filter image
    kernel = np.array([
        [-0.5, 0, 0.5],
        [-2, 0, 2],
        [-4, 0, 4],
        [-2, 0, 2],
        [-0.5, 0, 0.5]
    ])

    #Log filtering of image
    c = 255 / np.log(1 + np.max(gray)) 
    log_image = c * (np.log(gray + 1))
    log_image[np.isneginf(log_image)] = 255
    log_image = np.uint8(log_image)

    image_filtered = log_image

    upper_limit = 200
    lower_limit = 90

    ret, thresh1 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TOZERO_INV)

    thresh = np.concatenate((image_filtered, thresh1, thresh2, thresh3, thresh4, thresh5), axis=1)
    thresh = cv2.resize(thresh, (1400, 540), interpolation = cv2.INTER_AREA)

    cv2.imshow('Binary, Binary Inverted, Truncated, Zero, Zero Inverted Threshold', thresh)
    cv2.waitKey(0)


