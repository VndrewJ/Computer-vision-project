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

# files = [files[0]]

for file in files:
    fitem = os.path.join(inputPar, file)
    img = cv2.imread(fitem)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kernal used to filter image
    kernel = np.array([
        [0, -1, 0],
        [-1, 6, -1],
        [0, -1, 0]
    ])

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel_image = cv2.filter2D(blurred, -1, kernel)
    image_filtered = kernel_image

    # #Log filtering of image
    # c = 255 / np.log(1 + np.max(kernel_image)) 
    # log_image = c * (np.log(kernel_image + 1))
    # log_image[np.isneginf(log_image)] = 255
    # log_image = np.uint8(log_image)

    # image_filtered = log_image
    # image_filtered = cv2.equalizeHist(log_image)
    # image_filtered = cv2.Sobel(kernel_image, cv2.CV_64F,0,1,ksize=3) 
    # image_filtered = cv2.Canny(kernel_image, 50, 150, apertureSize=3)
    # exp_image = np.power(gray / 255.0, 5) * 255.0

    upper_limit = 200
    lower_limit = 110

    ret, thresh1 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TOZERO_INV)

    thresh = np.concatenate((blurred, image_filtered, thresh1, thresh2, thresh3, thresh4, thresh5), axis=1)
    thresh = cv2.resize(thresh, (1400, 540), interpolation = cv2.INTER_AREA)

    # cv2.imshow('Binary, Binary Inverted, Truncated, Zero, Zero Inverted Threshold', thresh)
    # cv2.waitKey(0)

    edges = thresh2

    # This returns an array of r and theta values
    lines_list =[]
    lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=10 # Max allowed gap between line for joining them
            )
 
    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1,y1),(x2,y2)])

    print_image = cv2.resize(img, (1400, 540), interpolation = cv2.INTER_AREA)
    cv2.imshow('test', print_image)
    cv2.waitKey(0)


