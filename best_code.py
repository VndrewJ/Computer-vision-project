# %%

import cv2
import os
import numpy as np
import csv
import math

# Function definitions

def find_weld_gap(height_index, line_image, main_image):
    weld_line = line_image[height_index, :]
    weld_positions = []
    for index in range(len(weld_line) - 1):
        if list(weld_line[index]) == [0,255,0]:
            weld_positions.append(index)
    
    cv2.line(main_image, (0, 70), (len(weld_line) - 1, 70), (255,0,0),5)
    if len(weld_positions) != 0:
        cv2.line(main_image, (min(weld_positions), 70), (max(weld_positions), 70), (0,0,255),10)
    
    return main_image, weld_positions

def save_image(file_name, image, interim_no, gap_position):
    # Saves image with filename format requested
    output_image = crop_image(image, gap_position)
    # If final image is desired, set interim_no to 0
    name_end = file.find(".")
    image_title = file_name[0:name_end]
    if interim_no == 0:
        output_name = f'{image_title.capitalize()}_A_WeldGapPosition.jpg'
    else:
        output_name = f'{image_title.capitalize()}_B_InterimResult{interim_no}.JPG '
    # Define the output file path
    fout = os.path.join(outPar, output_name)
    # Save the image with detected edges
    cv2.imwrite(fout, output_image)

def check_validity(weld_indices, last_gap_position):
    try:
        # check the distance of the weld indices
        width = max(weld_indices)- min(weld_indices)
        location = math.floor(width/2) + min(weld_indices)
        if (width <11 and len(weld_indices) >=3 and ((last_gap_position == 0) or abs(last_gap_position-location)<22)):
            # Check width is within tolerance and that hasnt moved more that 1mm from the last image
            # Also check that there is at least 3 measurements for the weld line
            last_gap_position = location
            return location, 1
        else:
            return -1, 0
    except ValueError:
        # No weld indices exist
        return -1, 0

def crop_image(image, gap_position):
    crop_width = 600
    crop_height = 400

    # If gap_position = 0, do not crop width-wise
    cols = np.array(image.shape[1])
    if gap_position ==0:
        return image[0:crop_height, 0:cols]
    # otherwise crop image so gap is in the center 
    else:
        left_index = gap_position-crop_width if gap_position >= crop_width else 0
        right_index = gap_position+crop_width if gap_position <= cols - crop_width else cols
        return image[0:crop_height, left_index:right_index]

#******************Main code start********************#

# Get the current working directory
path = os.getcwd()

# Define input and output directories
input_dir = os.path.join(path, 'image_datasets')
outPar = os.path.join(path, 'InterimResults')

os.makedirs(outPar, exist_ok=True)

#Initialise weld positions array and last logged weld position
weld_positions = []

#iterate through each set
for subdir in os.listdir(input_dir):
    inputPar = os.path.join(input_dir, subdir)

    # List all files in the input directory
    files = os.listdir(inputPar)

    # Reset last gap position between sets
    last_gap_position = 0

    #iterate through each image in a set
    for file in files:
        fitem = os.path.join(inputPar, file)
        img = cv2.imread(fitem)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #initialise convolution kernal for vertical lines
        kernel = np.array([
            [-0.8, 0, 0.8],
            [-1.5, 0, 1.5],
            [-0.8, 0, 0.8]
        ])

        #apply gaussian blur and kernal
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        kernel_image = cv2.filter2D(blurred, -1, kernel)
        image_filtered = kernel_image

        upper_limit = 255
        lower_limit = 70

        #apply threshold to filtered image
        ret, thresh1 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY)
        edges = thresh1

        # This returns an array of r and theta values to get the lines
        lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=100, # Min number of votes for valid line
                minLineLength=5, # Min allowed length of line
                maxLineGap=10 # Max allowed gap between line for joining them
                )

        # Draw the lines
        try:
            lines_list =[]
            for points in lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]
                # Draw the lines joing the points
                # On the original image
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                # Maintain a simples lookup list for points
                lines_list.append([(x1,y1),(x2,y2)])
        except TypeError:
            # No lines exist
            continue

        #find the size of the weld gap and check its validity
        weld_image, weld_indices = find_weld_gap(70, img, img)
        weld_location, weld_valid = check_validity(weld_indices, last_gap_position)
        if weld_location != -1:
            # Updates position of last gap if the last location was valid
            last_gap_position = weld_location

        #Add image, final index and validity of answer to array
        weld_positions.append([file, weld_location, weld_valid])

        # print_image = cv2.resize(img, (1400, 540), interpolation = cv2.INTER_AREA)

        # Saves requested images
        save_image(file, img, 0, last_gap_position)
        save_image(file, kernel_image, 1, last_gap_position)
        save_image(file, thresh1, 2, last_gap_position)
        # Shows image if uncommented
        # cv2.imshow('test', print_image)
        # cv2.waitKey(0)
        
#Output all results to csv file
with open('WeldGapPositions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image name', 'Weld Gap Position', 'Weld Gap Valid'])
        writer.writerows(weld_positions)




