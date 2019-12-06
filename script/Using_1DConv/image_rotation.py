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
       # _, frame = cap.read()
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

def circShifting(image, shift):
    height, width = image.shape[:2]
    T = np.float32([[1, 0, 0], [0, 1, shift]]) 
    temp = image[:][-shift:]
    check = cv2.warpAffine(image, T, (width, height))
    check[:][:shift] = temp
    return check

#############Data-Loading####################

#### Method1######
#pathname = "../data/fingers/modified/"
#dir_list = os.listdir(pathname)
#cv2.createTrackbar("L - V", "Trackbars", 84, 255, nothing)
pathname = "../../data/fingers/fingers/train/"
dir_list = os.listdir(pathname)

frame = cv2.imread(pathname + dir_list[12])
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("closed",closing)

    _,contours,_= cv2.findContours(closing.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    contours.sort( key = cv2.contourArea,reverse=True)
    c = contours[0]
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    kernel = np.ones((3,3),np.uint8)
    edge = cv2.Canny(closing,100,200)    
    edge = cv2.dilate(edge,kernel,iterations = 1)
    polar = cv2.linearPolar(edge, (cx, cy), 40, cv2.WARP_FILL_OUTLIERS)      
    cv2.imshow("polar",polar)
    
    ########## Linear Convolution in images ################
    #data gen
    center = (cx, cy)
    height, width = closing.shape[:2]
    scale = 1
    angle = 340  
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(closing, M, (width, height)) 
    cv2.imshow("rotated",rotated)   
    
    ## Polar conv
    kernel = np.ones((3,3),np.uint8)
    edge = cv2.Canny(rotated,100,200)    
    edge = cv2.dilate(edge,kernel,iterations = 1)
    rpolar = cv2.linearPolar(edge, (cx, cy), 40, cv2.WARP_FILL_OUTLIERS)      
    cv2.imshow("polar_rotated",rpolar)
    
   
    key = cv2.waitKey(1)
    if key == 27:
        break

 ##convolution
shift = 1
res = np.zeros((1,int(height/shift)))    
for i in range(int(height/shift)):
    temp = circShifting(rpolar,(i+1)*shift)
    res[0][i] = np.sum(np.multiply(polar,temp))

ans = np.where(res == np.amax(res))
print((ans[1]+1)*360/128)
##################Segmentation####################
#dist = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
#dist = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
#laplacian = cv2.Laplacian(dist,cv2.CV_64F)
#laplacian1 = laplacian/laplacian.max()
#cv2.imshow('a7',laplacian1)
#cv2.waitKey(0)

#plt.subplot(2,1,1)
#plt.imshow(dist)
#plt.subplot(2,1,2)
#plt.imshow(laplacian)



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
