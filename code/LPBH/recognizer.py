#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import sys
import cv2

# LBPH人脸识别
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 人脸检测
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 加载训练后的模型
recognizer.read('model.yml')
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

def usage():
    print ("usage():")
    print ("    {} name_file".format(__file__))
    print ("    {} name_file img_path".format(__file__))
    print ("")

# 读取人名列表
def load_namelist(file_path):
    if not os.path.exists(file_path):
        print("Name file does not exist!")
        exit()
    name_list = ["unknow"]
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            name = line.strip()
            name_list.append(name)
            line = fp.readline()
    return name_list

def recognize_from_camera(name_list):
    # Init Camera
    cap = cv2.VideoCapture(0)
    minW = 0.1 * cap.get(3)
    minH = 0.1 * cap.get(4)

    while cap.isOpened():
        # Capture images frame by frame
        isSuccess, frame = cap.read()
        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #识别人脸
        faces = face_detector.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW),int(minH))
                )
        for(x,y,w,h) in faces:
            # 圈选人脸
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # 人脸识别
            label_num, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # confidence为0的时候可信度最高
            # 这里我们认为confidence小于50都可接受
            if confidence < 50:
                name = name_list[label_num]
                possibility = "{}%",format(round(100-confidence))
            else:
                name = name_list[0]
                possibility = "{}%",format(round(confidence))
            # 显示姓名和概率
            cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255, 0, 0), 1)
            cv2.putText(frame, possibility ,(x+5, y+h-5), font, 1, (0, 255, 0), 1)
        #展示结果
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def recognize_from_image(name_list, img_path):
    if not os.path.exists(img_path):
        print("Picture file does not exist!")
        exit()
    img = cv2.imread(img_path)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #识别人脸
    faces = face_detector.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW),int(minH))
            )
    for(x,y,w,h) in faces:
        # 圈选人脸
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # 人脸识别
        label_num, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # confidence为0的时候可信度最高
        # 这里我们认为confidence小于50都可接受
        if confidence < 50:
            name = name_list[label_num]
            possibility = "{}%",format(round(100-confidence))
        else:
            name = name_list[0]
            possibility = "{}%",format(round(confidence))
        # 显示姓名和概率
        cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, possibility ,(x+5, y+h-5), font, 1, (0, 255, 0), 1)
    #展示结果
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    
if __name__ == "__main__":

    if len(sys.argv) == 2:
        name_file = sys.argv[1]
        name_list = load_namelist(name_file)
        recognize_from_camera(name_list)

    elif len(sys.argv) == 3:
        name_file = sys.argv[1]
        img_path = sys.argv[2]
        name_list = load_namelist(name_file)
        recognize_from_image(name_list, img_path):
            
    else:
        usage()
        exit()
