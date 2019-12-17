#!/usr/bin/env python3
# -*- coding: utf8 -*-
#
#    Copyright XiuzhengLi <xiuzhengli@qq.com>
#

import cv2
import dlib

# Init Camera
cap = cv2.VideoCapture(0)

detector =dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../TensorFace/openface/models/dlib/shape_predictor_68_face_landmarks.dat')

while cap.isOpened():
    # Capture images frame by frame
    isSuccess,frame = cap.read()

    # Display
    if isSuccess:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face
        dets = detector(img, 1)
    
        for k, d in enumerate(dets):
            shape = predictor(img, d)
        
            for index, pt in enumerate(shape.parts()):
                cv2.circle(frame, (pt.x, pt.y), 1, (255, 0, 0), 2)
        cv2.imshow("frame",frame)

    # Press "q" to exit
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

# Release Camera
cap.release()
cv2.destoryAllWindows()
