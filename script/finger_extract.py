
import cv2
import os
from matplotlib import image
import numpy as np

##################FUNCTIONS##################################

def nothing(x):
    pass


def thresh_visual(pathname):
    frame = cv2.imread(pathname)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - V", "Trackbars", 84, 255, nothing)

    while True:

        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        ret,thr = cv2.threshold(gray,l_v,255,cv2.THRESH_BINARY)
        cv2.imshow("frame",frame)
        cv2.imshow("thr",thr)

        key = cv2.waitKey(1)
        if key == 27:
            break

def store_data(pathname,destname):
    dir_list = os.listdir(pathname)
    kernel = np.ones((2,2),np.uint8)
    count = 0;
    for item in dir_list[10000:10400]:
        if item[-5] == "R":
            count +=1
            frame = cv2.imread(pathname + item)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret,thr = cv2.threshold(gray,84,255,cv2.THRESH_BINARY)
            thr = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,kernel)
            image.imsave( destname + item, thr)
            pass
    print(count)
#######################START##############################

pathname = "../data/fingers/fingers/train/"
destname = "../data/fingers/fingers/test_modified/"
store_data(pathname,destname)

###################################END######################
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
