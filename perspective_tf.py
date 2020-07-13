import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import detect_corners_from_contour, four_point_transform


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())


dir_ = args["images"]
for img_name in os.listdir(dir_):
    img = cv2.imread(dir_+img_name)
    img = cv2.resize(img, (256, 80))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5),0)

    # _, img_blur = cv2.threshold(img_blur, 250, 255, cv2.THRESH_OTSU)
    img_canny = cv2.Canny(img_blur, 70, 200)
    
    contours, _ = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img, cnt, -1, (255, 0, 0), 3) 
    corners = detect_corners_from_contour(img_gray, cnt)
    img_ptf = four_point_transform(img_gray, corners)

    fig = plt.figure(1, (6,3))
    
    ax = plt.subplot(1,3,1)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("original_img")

    ax = plt.subplot(1,3,2)
    ax.imshow(img_canny, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("canny_img")

    ax = plt.subplot(1,3,3)
    ax.imshow(img_ptf, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("perspective_img")

    plt.show()