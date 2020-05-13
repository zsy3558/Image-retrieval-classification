import time

import cv2
import numpy as np
import os
import object_detector

def main():
    img_path = "./0.jpg"
    image = cv2.imread(img_path)
    # video_path = input("Enter your video path: ")
    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    object_detector.get_profile(image)
    imgs_num = len(os.listdir("./data/"))+1
    cv2.imwrite("./data/"+str(imgs_num)+".jpg", image)

if __name__ == '__main__':
    main()
