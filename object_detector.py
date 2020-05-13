import cv2
import numpy as np
import shutil
import os

def pretreatment(image):
    if not os.path.exists("./detected_img/"):
        os.makedirs("./detected_img/")
    else:
        shutil.rmtree("./detected_img/")
        os.makedirs("./detected_img/")
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=0)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    save_name = "./detected_img/01.png"
    cv2.imwrite(save_name, gradient)
    blurred_img = cv2.blur(gradient, (16, 16))
    (_, Binary_img) = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
    erode_img = cv2.erode(Binary_img, None, iterations=4)
    dilate_img = cv2.dilate(erode_img, None, iterations=4)
    return Binary_img

def get_profile(image):
    img = pretreatment(image)
    copyImg = image.copy()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_low = np.array([0, 127, 127])
    color_high = np.array([127, 255, 255])
    inRange_value = cv2.inRange(hsv_img, color_low, color_high)
    hsv_img[inRange_value > 0] = ([255, 255, 255])

    ## converting the HSV image to Gray inorder to be able to apply contouring
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray_img = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

    _, Binary_img = cv2.threshold(gray_img, 90, 255, 0)

    contours, hierarchy = cv2.findContours(Binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(copyImg, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("./detected_img/copyImg.jpg", copyImg)

    (cnts, _) = cv2.findContours(Binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #for n in range(min(len(cnts), 5)):
    imgInfo = image.shape
    for n in range(len(cnts)):
        c = sorted_cnts[n]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        height = y2 - y1
        width = x2 - x1
        #if(height>=imgInfo[0]*0.1 or width>=imgInfo[1]*0.1):
        if (height*width >= imgInfo[0] * 0.1 * imgInfo[1] * 0.1):
            cropImg = image[y1:y1 + height, x1:x1 + width]
            save_name = "./detected_img/" + str(n) + ".jpg"
            cv2.imwrite(save_name, cropImg)