import cv2
from google.colab import drive
import glob
import numpy as np
#from matplotlib import pyplot as plt

drive.mount('/content/drive', force_remount = True)


def label_it(y_train,path):
    if path[-6] == "0":
        y_train.append([0,0,0,0,0,1])
    if path[-6] == "1":
        y_train.append([0,0,0,0,1,0])
    if path[-6] == "2":
        y_train.append([0,0,0,1,0,0])
    if path[-6] == "3":
        y_train.append([0,0,1,0,0,0])
    if path[-6] == "4":
        y_train.append([0,1,0,0,0,0])
    if path[-6] == "5":
        y_train.append([1,0,0,0,0,0])
    

img_path = glob.glob("/content/drive/My Drive/modified/*.png")
y_train = []
x_train = []

for path in img_path:
  image = cv2.imread(path)
  x_train.append(image)
  label_it(y_train,path)
  

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train_gr = np.zeros([len(x_train),128,128])
for k in range(len(x_train)):
  temp = x_train[k]
  temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
  temp[temp == 14] = 0 #corresponding during image storing
  temp[temp == 38] = 1
  x_train_gr[k] = temp