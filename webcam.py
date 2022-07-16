import os
import cv2

cap = cv2.VideoCapture(0)
classifier_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
face_detect = cv2.CascadeClassifier(classifier_file)

while(True):
    # read
    ret, frame = cap.read()

    # safe
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else :
        break

    # detect face
    face = face_detect.detectMultiScale(rgb)

    # rectangle face
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow('frame', frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out = cv2.imwrite('capture.jpg', frame)
        break

cap.release()