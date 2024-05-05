import cv2
import os
import numpy as np
import csv

# Get the current working directory
path = os.getcwd()

# Define input and output directories
inputPar = os.path.join(path, 'image_datasets/Set 2/')
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

    width, height = img.shape[1::-1]

    img = img[0:400, 0:width]


    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([
        [-1, 0, 1],
        [-1.5, 0, 1.5],
        [-1, 0, 1]
    ])

    kernel = np.array([
        [-1, 1, 1],
        [-1, -1, 1],
        [-1, -1, 1]
    ])

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
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

    upper_limit = 255
    lower_limit = 80

    ret, thresh1 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image_filtered, lower_limit, upper_limit, cv2.THRESH_TOZERO_INV)

    thresh = np.concatenate((blurred, image_filtered, thresh1, thresh2, thresh3, thresh4, thresh5), axis=1)
    thresh = cv2.resize(thresh, (1400, 540), interpolation = cv2.INTER_AREA)

    # cv2.imshow('Thresholded images', thresh)
    # cv2.waitKey(0)

    edges = thresh1
 
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
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    os.remove(fitem)

