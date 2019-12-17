#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Copyright XiuzhengLi <xiuzhengli@qq.com>
#
# Development environment: RaspberryPi 4B, Python3.7, OpenCV4.1.0

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import dlib
import numpy

# Init the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(640, 480))
fps = 0

# load predictor
detector =dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/pi/Python/TensorFace/openface/models/dlib/shape_predictor_68_face_landmarks.dat')

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    img = frame.array
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect face
    dets = detector(img, 1)
    
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        
        for index, pt in enumerate(shape.parts()):
            cv2.circle(img, (pt.x, pt.y), 1, (255, 0, 0), 2)
    
    # Calculate and show the FPS
    fps = fps + 1

    cv2.imshow("Frame", img)
    cv2.waitKey(1)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)
