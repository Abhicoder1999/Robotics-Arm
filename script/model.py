import cv2
from google.colab import drive
import glob

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