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

# Loop through each file in the input directory
for file in files:
    fitem = os.path.join(inputPar, file)
    
    img = cv2.imread(fitem)

    # y=600
    # x=0
    # h=1300
    # w=4000
    # img_cropped = img[x:w, y:h]

    # img = img_cropped

    # cv2.imshow('something', img_cropped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # break

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kernal used to filter image
    kernel = np.array([
        [-0.5, 0, 0.5],
        [-2, 0, 2],
        [-4, 0, 4],
        [-2, 0, 2],
        [-0.5, 0, 0.5]
    ])
    # Apply gaussian blur
    # blurred = cv2.equalizeHist(gray) 
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    image_filtered = cv2.filter2D(blurred, -1, kernel)

    ret, image_filtered = cv2.threshold(image_filtered, 80, 255, cv2.THRESH_BINARY)
 
    # Apply edge detection method on the image
    edges = cv2.Canny(image_filtered, 50, 150, apertureSize=3)
 
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

    # Define the output file path
    fout = os.path.join(outPar, file)

    # Save the grayscale image with detected edges
    cv2.imwrite(fout, img)


files = os.listdir(outPar)

# Prints out files and deletes them consequtively
for file in files:

    fitem = os.path.join(outPar, file)
    
    img = cv2.imread(fitem)

    print_image = np.concatenate((edges, image_filtered, gray, blurred), axis = 1)
    print_image = cv2.resize(print_image, (1400, 540), interpolation = cv2.INTER_AREA)
    cv2.imshow('test', print_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    os.remove(fitem)
