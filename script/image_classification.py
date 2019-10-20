# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import os
from skimage import io
import matplotlib as plt
import cv2
import numpy as np

model = load_model("model2.h5")
#model.summary()

pathname = "../data/fingers/fingers/test_modified/"
dir_list = os.listdir(pathname)
index = 0
frame = cv2.imread( pathname + dir_list[index])
print(dir_list[index][-6:-4])

#while True:
#   frame = cv2.imread( pathname + dir_list[100])
#    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    temp[temp == 14] = 0 #corresponding during image storing
#    temp[temp == 38] = 1
#    
#    cv2.imshow("frame",frame)
#    key = cv2.waitKey(1)
#    if key == 27:
#        break
    
frame = cv2.imread( pathname + dir_list[110])
frame = np.array(frame)

temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
temp[temp == 14] = 0 #corresponding during image storing
temp[temp == 38] = 1


temp = np.expand_dims((temp), axis=3)
print(temp.shape)
temp = np.expand_dims((temp), axis=0)
print(temp.shape)

predictions = model.predict(temp)

print(predictions)
print(np.argmax(predictions[0]))

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
