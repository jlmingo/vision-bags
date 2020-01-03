##This file contains handy functions to be used in the main program
#Below are the packages needed to execute the program
from skimage.measure import compare_ssim
# import imutils
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
# import glob
# from scipy.spatial import distance

def showImg(nameimg, image, scale_percent=100):
    '''
    Shows image in screen, optional scale percent parameter (default=100% that is original image)
    '''
    while True:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow(nameimg, image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break

def contourObj(gray_orig, gray_mod):
    '''
    inputs:
        a) gray_orig is the background with no objects captured by the camera. It should always be the same photo.
        b) gray_mod is the foto to be analyzed
    outputs:
        A tuple consisting of:
            a) returning_values: a dictionary containing the marked image (the center and the borders are drawn on the original image)
            b) centroid: coordinates of the located centroid
    TO-DO: calculate and return the angle of the bag
    '''
    (score, diff) = compare_ssim(gray_orig, gray_mod, full=True)
    diff = (diff * 255).astype("uint8") #diff is the difference of the two inputs, useful to remark where the bag is
    thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #thresholds could be changed if others are considered more interesting
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #finds all contours on thesholded diff image
    area = []
    #Create list of areas for each contour
    for c in cnts:
        area.append(cv2.contourArea(c))
    #locates index for biggest area among all found
    max_cnt = area.index(max(area))
    #Approximate contour with highest area to polygon, this will give the desired perimeter
    epsilon = 0.05*cv2.arcLength(cnts[max_cnt],True)
    approx = cv2.approxPolyDP(cnts[max_cnt],epsilon,True)
    #Moments are some specific properties of a contour. We can calculate centroid using this input.
    M = cv2.moments(cnts[max_cnt])
    centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    #Draw contour and centroid on the image
    marked = cv2.drawContours(gray_mod, [approx], 0, (0,255,0), 3)
    cv2.circle(marked, centroid, 5, (0,255,0), 3)
    returning_values = {
        "marked_img": marked,
        "centroid": centroid
    }
    return returning_values, approx, thresh