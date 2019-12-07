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
    
def preProcess(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return closing
    
def findCentroid(image):
    _,contours,_= cv2.findContours(image.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    contours.sort( key = cv2.contourArea,reverse=True)
    c = contours[0]
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx,cy)

def polarConv(image,kernel,center):
    edge = cv2.Canny(image,100,200)    
    edge = cv2.dilate(edge,kernel,iterations = 1)
    polar = cv2.linearPolar(edge, center, 40, cv2.WARP_FILL_OUTLIERS)      
    return polar
    
def rotateImg(image,center,angle,scale):
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (width, height))
    return rotated

def circConv(polar,rpolar,height,shiftkey):
    res = np.zeros((1,int(height/shiftkey)))    
    for i in range(int(height/shiftkey)):   
        temp = circShifting(rpolar,(i+1)*shiftkey)
        res[0][i] = np.sum(np.multiply(polar,temp))
    return res


#############Data-Loading####################

#### Method1######
#pathname = "../data/fingers/modified/"
#dir_list = os.listdir(pathname)
#cv2.createTrackbar("L - V", "Trackbars", 84, 255, nothing)
pathname = "../../data/fingers/fingers/train/"
dir_list = os.listdir(pathname)

frame = cv2.imread(pathname + dir_list[14])
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
    closing = preProcess(frame)
    cv2.imshow("closed",closing)
    
    center = findCentroid(closing.copy())
    kernel = np.ones((3,3),np.uint8)    
    polar = polarConv(closing,kernel,center)
    cv2.imshow("polar",polar)
    
    #data gen
    scale = 1
    angle = 340  
    rclosing = rotateImg(closing,center,angle,scale)
    cv2.imshow("rotated",rclosing)
    rpolar = polarConv(rclosing,kernel,center)    
    cv2.imshow("polar_rotated",rpolar)
    
   
    key = cv2.waitKey(1)
    if key == 27:
        break

 ##convolution
shiftkey = 1 #no of gaps between conv)
res = circConv(polar,rpolar,height,shiftkey)
ans = np.where(res == np.amax(res))
print((ans[1]+1)*360/128)


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
