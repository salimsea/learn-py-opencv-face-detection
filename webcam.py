import os
import cv2

# cap = cv2.VideoCapture('rtsp://admin:YHWMZM@192.168.1.3/mpeg4')
cap = cv2.VideoCapture(0)
classifier_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
classifier_file_eye = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
# classifier_file = os.path.join(cv2.data.haarcascades, "haarcascade_fullbody.xml")
face_detect = cv2.CascadeClassifier(classifier_file)
eye_detect = cv2.CascadeClassifier(classifier_file_eye)

while(True):
    # read
    ret, frame = cap.read()
    resize = cv2.resize(frame, (400, 200)) 
    # resize = frame

    # safe
    if ret:
        rgb = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    else :
        break

    # detect face
    face = face_detect.detectMultiScale(rgb)
    eye = eye_detect.detectMultiScale(rgb)

    # rectangle face
    for (x,y,w,h) in face:
        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),2)
    for (x,y,w,h) in eye:
        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow('frame', resize)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out = cv2.imwrite('capture.jpg', resize)
        break

cap.release()