#!/usr/bin/env python3
# -*- coding: utf8 -*-
#
#    Copyright XiuzhengLi <xiuzhengli@qq.com>
#

import cv2

# Load classifier
frontalface_classifier = cv2.CascadeClassifier('../../opencv/data/haarcascades/haarcascade_frontalface_default.xml')
profileface_classifier = cv2.CascadeClassifier('../../opencv/data/haarcascades/haarcascade_profileface.xml')

# Init Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture images frame by frame
    isSuccess,frame = cap.read()

    # Display
    if isSuccess:
        frontal_faces = frontalface_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in frontal_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        profile_faces = profileface_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in profile_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("frame",frame)

    # Press "q" to exit
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

# Release Camera
cap.release()
cv2.destoryAllWindows()
