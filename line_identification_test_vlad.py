# %%

import cv2
import os
import numpy as np

# Get the current working directory
path = os.getcwd()

# Define input and output directories
inputPar = os.path.join(path, 'image_datasets/Set 1/')
outPar = os.path.join(path, 'output_images/')

os.makedirs(outPar, exist_ok=True)

# List all files in the input directory
files = os.listdir(inputPar)

# Loop through each file in the input directory
for file in files:
    fitem = os.path.join(inputPar, file)
    
    img = cv2.imread(fitem)
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Kernal used to filter image
    kernel = np.array([
        [-0.5, 0, 0.5],
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
        [-0.5, 0, 0.5]
    ])

    image_filtered = cv2.filter2D(img, -1, kernel)
    ret, image_filtered = cv2.threshold(image_filtered, 140, 255, cv2.THRESH_BINARY)
 
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

    cv2.imshow('test', img)

    cv2.waitKey(100)
    cv2.destroyAllWindows()
    os.remove(fitem)
