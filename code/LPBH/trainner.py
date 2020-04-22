#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import cv2
import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder

# LBPH人脸识别
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 人脸检测
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_data(data_path):
    img_list = []
    label_list = []
    img_paths = [os.path.join(path,f) for f in os.listdir(path)]
    for img_path in img_paths:
        label = os.path.split(image_path)[-1].split("_")[0]
        # 加载图片
        img = cv2.imread(img_file)
        # 图片灰度化
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for(x,y,w,h) in faces:
            img_list.append(img[y:y+h,x:x+w])
            label_list.append(label)
    return img_list, label_list

dataset = sys.argv[1]
imgs, labels = load_data(dataset)
label_encoder = LabelEncoder()
labels_num = label_encoder.fit_transform(data)
# 训练模型
recognizer.train(imgs, labels_num)
# 保存模型
recognizer.save('model.yml')
