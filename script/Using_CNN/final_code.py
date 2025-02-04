# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:51:09 2019

@author: abhijeet
"""

import cv2
import keras
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
from skimage import io, transform

def nothing(x):
    pass

model = load_model("model2.h5")
kernel = np.ones((3,3),np.uint8)
#lower_blue = np.array([0, 0, 0])
#upper_blue = np.array([0, 0, 0])


#################Function##############
def HSV_setting():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 84, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        cv2.imshow("mask", closing)
        values = np.array([lower_blue,upper_blue])


        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return values

def smoothing (img):
    #removing the distortions and noise
    kernel = np.ones((5,5),np.uint8)
    smooth_img = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel)
    # smooth_img =cv2.morphologyEx(smooth_img.copy(),cv2.MORPH_CLOSE,kernel)
    return smooth_img

def edgeDetection(img):
    # Canny edge detection after threshold and smoothing
    l_thr = 0#cv2.getTrackbarPos("L - Thr", "Canny_Parameter") #230
    h_thr = 179#cv2.getTrackbarPos("H - Thr", "Canny_Parameter") #280
    edges = cv2.Canny(img,l_thr,h_thr)
    return edges

def lineDetection (edges, img):
        #Line Detection For Algo and Case Study
        # rho = cv2.getTrackbarPos("rho", "Parameters")
        lines = cv2.HoughLinesP(edges,2,np.pi/180,56,minLineLength = 51,maxLineGap = 300)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(img,((x1+x2)/2,(y1+y2)/2), 3, (0,0,255), -1)
        return lines,img

#############Data-Loading####################
#pathname = "../data/fingers/modified/"
#dir_list = os.listdir(pathname)
#cv2.createTrackbar("L - V", "Trackbars", 84, 255, nothing)

#pathname = "../data/fingers/fingers/train/"
#dir_list = os.listdir(pathname)

#frame = cv2.imread(pathname + dir_list[110])

################# CALLIBRARTION ####################
#temp = HSV_setting()
#lower_blue = temp[0]
#upper_blue = temp[1]

lower_blue = np.array([36, 28, 0])
upper_blue = np.array([113, 212, 114])

############### PROCESSING ##############################
kernel = np.ones((3,3),np.uint8)
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("closed",closing)


    key = cv2.waitKey(1)
    if key == 27:
        break

    elif key == 113:
        mg_read = transform.resize(closing, (128,128), mode = 'constant')
        
        #print(mg_read.shape)
        temp = np.expand_dims((mg_read), axis=3)
        print(temp.shape)
        temp = np.expand_dims((temp), axis=0)
        print(temp.shape)
        
        predictions = model.predict(temp)
        
        print(predictions)
        print(np.argmax(predictions[0]))
        print("working")
        
        continue

##################Segmentation####################
#dist = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
#dist = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)m
#plt.imshow(dist)
    # ind = np.unravel_index(np.argmax(dist, axis=Nomask = cv2.inRange(hsv, lower_blue, upper_blue)ne), dist.shape)
    # ret, mask = cv2.threshold(dist,0.7*dist.max(),255,0)

#kernel = np.ones((5,5),np.uint8)
#mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    # cv2.imshow(mask)

#now find the mask angle using hough line estimation

#lines = cv2.HoughLinesP(mask,2,np.pi/180,56,minLineLength = 51,maxLineGap = 300)
#if lines is not None:
#    for line in lines:
#        x1,y1,x2,y2 = line[0]
#        cv2.line(gray,(x1,y1),(x2,y2),(0,255,0),2)
#        cv2.circle(gray,((x1+x2)/2,(y1+y2)/2), 3, (0,0,255), -1)

cap.release()
cv2.destroyAllWindows()

cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
