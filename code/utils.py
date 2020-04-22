#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Some basic functions
"""

import os
import sys
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# 定义常量
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# 加载分类器
frontalface_classifier = cv2.CascadeClassifier('./classifier/haarcascade_frontalface_default.xml')
profileface_classifier = cv2.CascadeClassifier('./classifier/haarcascade_profileface.xml')

# 读取文件
def readAllLines(file_path):
    all_lines = []
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            all_lines.append(line)
            line = fp.readline()
    return all_lines

# 图片采集
def get_pictures(output_dir, label):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 相机初始化
    cap = cv2.VideoCapture(0)
    index = 1
    while True:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('s'):
            file_name = label + '_' + str(index) + '.jpg'
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, frame)
            index += 1
    cap.release()
    cv2.destroyAllWindows()

# 人脸检测
def face_detection(img):
    minH = int(img.shape[0] * 0.2)
    minW = int(img.shape[1] * 0.2)
    # 识别正脸
    frontal_faces = frontalface_classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(minH, minW))
    for (x, y, w, h) in frontal_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 识别侧脸
    profile_faces = profileface_classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(minH, minW))
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# one hot 编码 / 解码
def one_hot_encode(data):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    one_hot = to_categorical(integer_encoded)
    return one_hot

def one_hot_decode(one_hot):
    data = np.argmax(one_hot)
    return data

# 图片中心crop
def image_crop(src_img):
    height = src_img.shape[0]
    width = src_img.shape[1]
    min_side = min(height, width)
    dl = min_side / 2
    dst_img = src_img[int(height / 2 - dl):int(height / 2 + dl), int(width / 2 - dl):int(width / 2 + dl)]
    return dst_img
 
# 图片大小resize
# INTER_NEAREST 最近邻插值
# INTER_LINEAR  双线性插值
def image_resize(src_img):
    dst_img = cv2.resize(src_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return dst_img

# 图片边界填充0
def image_pad(src_img):
    height = src_img.shape[0]
    width = src_img.shape[1]
    max_side = max(height, width)
    dh = int((max_side - height) / 2)
    dw = int((max_side - width) / 2)
    dst_img = cv2.copyMakeBorder(src_img, dh, dh, dw, dw, cv2.BORDER_CONSTANT,value=[0,0,0])
    return dst_img

# if __name__ == "__main__":
#     get_pictures("/home/lxzheng/test_data", "lxzheng")
