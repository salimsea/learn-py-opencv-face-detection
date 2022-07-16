import os
import cv2

cv2.VideoCapture(0)

trained_face_data = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"))

img = cv2.imread('salim.jpeg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles around the faces
(x,y,w,h)=face_coordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)

cv2.imshow('akwoakwo', img)
cv2.waitKey()



print("Code Successfully")