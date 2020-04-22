#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import sys
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

def usage():
    print ("usage():")
    print ("    {} output_dir label".format(__file__))
    print ("")
    print ("    Press s to save the pictures")
    print ("    Press q to exit")
    print ("")

# 图片采集
def get_pictures(output_dir, label):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 相机初始化
    cap = cv2.VideoCapture(0)
    index = 1
    while True:
        ret, frame = cap.read()
        show_frame = frame
        # cv2.putText(show_frame, "Press \"s\" to save the pictures", (30, 30), font, 1, (0, 0, 255), 2)
        # cv2.putText(show_frame, "Press \"q\" to exit", (30, 60), font, 1, (255, 0, 0), 2)
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        exit()
    else:
        output_dir = sys.argv[1]
        label = sys.argv[2]
        get_pictures(output_dir, label)
