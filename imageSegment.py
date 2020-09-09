# -*- coding: utf-8 -*-
"""
imageSegment.py

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image

    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filter_img = cv2.medianBlur(original_img,9)
    hsv_img = cv2.cvtColor(filter_img, cv2.COLOR_RGB2HSV)
    
    green_low = np.array([45, 40, 40])
    green_high = np.array([75,255,255])
    green_mask = cv2.inRange(hsv_img, green_low, green_high) 
    
    yellow_low = np.array([15, 40, 40])
    yellow_high = np.array([45,255,255])
    yellow_mask = cv2.inRange(hsv_img, yellow_low, yellow_high)
    
    red_low = np.array([1, 40, 40])
    red_high = np.array([15,255,255])
    red_mask = cv2.inRange(hsv_img, red_low, red_high)
    red_low2 = np.array([173, 40, 40])
    red_high2 = np.array([179,255,255])
    red_mask2 = cv2.inRange(hsv_img, red_low2, red_high2)
    
    outImg = green_mask + yellow_mask + red_mask + red_mask2
    outImg = np.clip(outImg,0,1)   

    return outImg
