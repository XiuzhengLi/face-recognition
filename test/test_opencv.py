#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Copyright XiuzhengLi <xiuzhengli@qq.com>
#
# The simple test of opencv and camera.
# Development environment: Python3.7, OpenCV4.1.0

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

# Init the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(640, 480))

fps = 0

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array

    # Calculate and show the FPS
    fps = fps + 1

    cv2.imshow("Frame", image)
    cv2.waitKey(1)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)
