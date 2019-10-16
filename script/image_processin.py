import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

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
pathname = "../data/fingers/modified/"
dir_list = os.listdir(pathname)
frame = cv2.imread(pathname + dir_list[0])
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray[gray == 14] = 0
gray[gray == 38] = 1


#kernel = np.ones((20,20),np.uint8)
#gray = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel) 


#plt.imshow(gray)

##################Segmentation####################
dist = cv2.distanceTransform(gray, cv2.DIST_L2, 3)
dist = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
#plt.imshow(dist)
ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
ret, mask = cv2.threshold(dist,0.7*dist.max(),255,0)

#kernel = np.ones((5,5),np.uint8)
#mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)  
plt.imshow(mask)

#now find the mask angle using hough line estimation

lines = cv2.HoughLinesP(mask,2,np.pi/180,56,minLineLength = 51,maxLineGap = 300)
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(gray,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(gray,((x1+x2)/2,(y1+y2)/2), 3, (0,0,255), -1)


cv2.waitKey(0)
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