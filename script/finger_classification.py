 
import cv2


##################FUNCTIONS##################################

def nothing(x):
    pass


def thresh_visual():
    frame = cv2.imread("../data/fingers/train/0a2e7a71-e702-4f3d-9add-282d38163277_2L.png")    
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
#######################START##############################








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