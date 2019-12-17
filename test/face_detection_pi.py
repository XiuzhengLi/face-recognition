#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Copyright XiuzhengLi <xiuzhengli@qq.com>
#
# Development environment: RaspberryPi 4B, Python3.7, OpenCV4.1.0

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

# Init the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(640, 480))
fps = 0

# Load classifier
frontalface_classifier = cv2.CascadeClassifier('/home/pi/Python/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
profileface_classifier = cv2.CascadeClassifier('/home/pi/Python/opencv/data/haarcascades/haarcascade_profileface.xml')

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
    
    frontal_faces = frontalface_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in frontal_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    profile_faces = profileface_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Calculate and show the FPS
    fps = fps + 1

    cv2.imshow("Frame", image)
    cv2.waitKey(1)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)
