import os
import cv2
import math
import argparse
import numpy as np
from deskew import determine_skew
from matplotlib import pyplot

from utils import rotate_image


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())

dir_ = args["images"]

for img_name in os.listdir(dir_):
    img = cv2.imread(dir_+img_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256,80))

    aligned_img = img.copy()
    try:
        angle = determine_skew(img)
        if abs(angle) < 20:
            aligned_img = rotate_image(img, angle, (0, 0, 0))
    except Exception as e:
        pass

    pyplot.figure(1, (6,2))
    ax = pyplot.subplot(1,2,1)
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("image")
    ax = pyplot.subplot(1,2,2)
    ax.imshow(aligned_img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("aligned image")
    pyplot.show()