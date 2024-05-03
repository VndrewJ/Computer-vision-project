import matplotlib.pyplot as plt
import numpy as np
import cv2

# Draw the lines represented in the hough accumulator on the original image
def drawhoughLinesOnImage(image, houghLines):
    for line in houghLines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0), 2)   

# Different weights are added to the image to give a feeling of blending
def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta,gamma)

# Finds indices of weld gap and draws this on the main image
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

image = cv2.imread("image_datasets/Set 1/image0001.jpg") # load image in grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurredImage = cv2.GaussianBlur(gray_image, (5, 5), 0)
edgeImage = cv2.Canny(blurredImage, 50, 120)

# Detect points that form a line
dis_reso = 1 # Distance resolution in pixels of the Hough grid
theta = np.pi /180 # Angular resolution in radians of the Hough grid
threshold = 170# minimum no of votes

houghLines = cv2.HoughLines(edgeImage, dis_reso, theta, threshold)

houghLinesImage = np.zeros_like(image) # create and empty image

drawhoughLinesOnImage(houghLinesImage, houghLines) # draw the lines on the empty image
orginalImageWithHoughLines = blend_images(houghLinesImage,image) # add two images together, using image blending
weld_image, weld_indices = find_weld_gap(70, houghLinesImage, orginalImageWithHoughLines)

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
# ax1.imshow(image)
# ax1.set_title('Original Image')
# ax1.axis('off')

# ax2.imshow(edgeImage, cmap='gray')
# ax2.set_title('Edge Image')
# ax2.axis('off')

# ax3.imshow(orginalImageWithHoughLines, cmap='gray')
# ax3.set_title("Original Image with Hough lines")
# ax3.axis('off')

# cv2.imshow('test', image)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# cv2.imshow('test', edgeImage)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# cv2.imshow('test', orginalImageWithHoughLines)
# cv2.waitKey(0)

cv2.imshow('test', weld_image)
cv2.waitKey(0)

cv2.destroyAllWindows()