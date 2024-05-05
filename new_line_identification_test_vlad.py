import cv2
import os
import numpy as np
import csv

# Get the current working directory
path = os.getcwd()

# Define input and output directories
inputPar = os.path.join(path, 'image_datasets/Set 3/')
outPar = os.path.join(path, 'output_images/')

os.makedirs(outPar, exist_ok=True)
# List all files in the input directory
files = os.listdir(inputPar)
# Loop through each file in the input directory

i = 1


weld_positions = []

for file in files:
    fitem = os.path.join(inputPar, file)

    img = cv2.imread(fitem)

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kernal used to filter image
    kernel = np.array([
        [-0.5, 0, 0.5],
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
        [-0.5, 0, 0.5]
    ])

    l = 10

    # kernel = np.array([
    #     [-l, l, l],
    #     [-l, -l, l],
    #     [-l, -l, l]
    # ])

    image_filtered = cv2.filter2D(img, -1, kernel)

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
    
    def save_image(file_name, image, interim_no):
        # Saves image with filename format requested
        # If final image is desired, set interim_no to 0
        name_end = file.find(".")
        image_title = file_name[0:name_end]
        if interim_no == 0:
            output_name = f'{image_title.capitalize()}_A_WeldGapPosition.jpg'
        else:
            output_name = f'{image_title.capitalize()}_B_InterimResult{interim_no}.JPG '
        # Define the output file path
        fout = os.path.join(outPar, output_name)
        # Save the grayscale image with detected edges
        cv2.imwrite(fout, image)

    weld_image, weld_indices = find_weld_gap(70, img, img)

    #Add image, final index and validity of answer to array
    weld_positions.append([file, weld_indices, 1])

    print_image = cv2.resize(img, (1400, 540), interpolation = cv2.INTER_AREA)

    save_image('output_images', print_image, i)

    i = i + 1




files = os.listdir(outPar)
# Prints out files and deletes them consequtively

#Output all results to csv file
with open('WeldGapPositions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image name', 'Weld Gap Position', 'Weld Gap Valid'])
        writer.writerows(weld_positions)

for file in files:
    fitem = os.path.join(outPar, file)
    
    img = cv2.imread(fitem)

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove(fitem)

