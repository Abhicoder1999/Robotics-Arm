# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:31:10 2020

@author: abhijeet
"""

import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

def nothing(x):
    pass

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
        frame = cv2.imread("../data/fingers/fingers/train/0a2e7a71-e702-4f3d-9add-282d38163277_2L.png")
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
        cv2.imshow("mask", mask)
        values = np.array([lower_blue,upper_blue])


        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return values
    
def preProcess(image):
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return closing
    
def findCentroid(image,RGB):
    try:
        _,contours,_= cv2.findContours(image.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        contours.sort( key = cv2.contourArea,reverse=True)
        c = contours[0]
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #param = cv2.minEnclosingCircle(c)
        param = (cx,cy)
    except ValueError:
        print("no contours")
    except IndexError:
        print("no contours")
    except ZeroDivisionError:
        print("no contours")
    return (param)



#############Data-Loading####################

#### Method1######
#pathname = "../data/fingers/modified/"
#dir_list = os.listdir(pathname)
#cv2.createTrackbar("L - V", "Trackbars", 84, 255, nothing)
pathname = "../../data/fingers/fingers/train/"
dir_list = os.listdir(pathname)

frame = cv2.imread(pathname + dir_list[25])
#lower_blue = np.array([86, 45, 36]) #pic
#upper_blue = np.array([122, 255, 215]) #pic
lower_blue = np.array([0, 0, 84])
upper_blue = np.array([179, 255, 255])

#####Method2#########

#temp = HSV_setting()
#lower_blue = temp[0]
#upper_blue = temp[1]

kernel = np.ones((3,3),np.uint8)
cap = cv2.VideoCapture(0)

############ Algorithm #####################
while True:
    #_,frame = cap.read()
    height, width = frame.shape[:2]
    closing = preProcess(frame.copy())
    param = findCentroid(closing,frame)
    print(param)
    #frame = cv2.circle(frame.copy(),(int(param[0][0]),int(param[0][1])),int(param[1]), (0,0,255), 1)
    #cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)

    cv2.imshow("RGB",frame)
    cv2.imshow("closing",closing)
    

    key = cv2.waitKey(1)
    if key == 27:
        break


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
cv2.waitKey(1)
cv2.waitKey(1)
