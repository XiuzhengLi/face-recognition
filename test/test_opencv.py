#!/usr/bin/env python3
# -*- coding: utf8 -*-
#
#    Copyright XiuzhengLi <xiuzhengli@qq.com>
#

import cv2

# Init Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture images frame by frame
    isSuccess,frame = cap.read()

    # Display
    if isSuccess:
        cv2.imshow("frame",frame)

    # Press "q" to exit
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

# Release Camera
cap.release()
cv2.destoryAllWindows()
