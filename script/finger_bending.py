#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:39:32 2018

@author: abhijeet
"""
import cv2
import numpy as np
import time
import serial
import math

area_thr = 2500
prv_center = np.zeros(5)
prv_count = 0
time_thr = 30
time_elapsed = 0
diff_thr = 50

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)
ser = serial.Serial('/dev/ttyACM0',9600)

def send(i):
	if i == 0:
		ser.write('a')
	if i == 1:
		ser.write('b')
	if i == 2:
		ser.write('c')
	if i == 3:
		ser.write('d')
	if i == 4:
		ser.write('e')
	if i == 6:
		ser.write('r')

	

def serialCall(f):
	ser.write('s')
	for i in range(5):
		if f[i] == 0:
			print(i)
			send(i)
	ser.write('z')				
	

while True:
    _,frame = cap.read();
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = np.array([115,0,101])
    upper_red = np.array([179,255,255])
    #logic
    mask = cv2.inRange(hsv, lower_red, upper_red)
    closing = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    _,contours,_= cv2.findContours(closing.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = np.zeros(5)
    count = 0
    
    try:
	#print("try")
	contours.sort( key = cv2.contourArea,reverse=True)	
	for i in range(5):
		#print("range")
    		c = contours[i]
		area = cv2.contourArea(c)
		if area > area_thr:
			#print("area_thr")
    			M = cv2.moments(c)
    			cx = int(M['m10']/M['m00'])
    			cy = int(M['m01']/M['m00'])
			center[count] = cx
			
			count = count + 1	    			
			cv2.circle(frame,(cx,cy),6,(0,0,255),-1)
    			cv2.drawContours(frame, c, -1, (0,255,0), 3)
			if prv_center[0] != 0.0:
				for j in range(4):
					avg_center = (int)(math.floor((prv_center[j+1]+prv_center[j])/2))
					cv2.line(frame,(avg_center,0),(avg_center,511),(255,0,0),2)
	print("for out",count,prv_count)	
	if count == 5:
		prv_center = center;
		print("5 fingers detected",count,prv_count)
		prv_center.sort()
		send(6)
		print(6)
	

	if count < 5 and count == prv_count:
		print("less than 5 ",count,prv_count," ",time_elapsed)
		time_elapsed = time_elapsed+1
			
		if time_elapsed > time_thr:
			finger = {0:0,1:0,2:0,3:0,4:0}
			center.sort()
			print(center,prv_center)	
			for i in range(5):#
				diff = [0,0,0,0,0]
				temp = 0
				for j in range(5):
					diff[temp] = abs(prv_center[j] - center[i])
					temp = temp+1			
				minpos = diff.index(min(diff))
				print(center[i],diff,minpos)
				if diff[minpos] <= diff_thr:				
					finger[minpos] = 1
			print(finger)
			serialCall(finger)
			time_elapsed = 0
		
	if count>prv_count or count<prv_count:
		time_elapsed = 0
		#print("else case")		
		prv_count = count	
	
	prv_count = count
	print("prv_update",prv_count)
		
    except ValueError: 	
	cv2.imshow("frame",frame)
    except IndexError:
	cv2.imshow("frame",frame)
    except ZeroDivisionError:
	println("no contours") 
	cv2.imshow("frame",frame)

    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break;

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
cv2.waitKey(1)

