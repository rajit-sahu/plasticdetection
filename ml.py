from rembg import remove
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

cam = cv2.VideoCapture(0)
time.sleep(2)
result, imt = cam.read()
out = remove(imt)
cv2.imwrite('new.jpg',out)
cam.release()

imt = cv2.imread('new.jpg')
imt=cv2.resize(imt,(224,224))
new1 = []
new1.append(imt)
imtest=np.array(new1)
imtest = imtest/255.0
model = load_model("model.h5")
a=model.predict(imtest)
pr = round(a[0][0])
if(pr==0):
    print("plastic")
elif(pr==1):
    print("non-plastic")
else:
    print("error")