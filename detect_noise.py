import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot
from skimage.restoration import estimate_sigma


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())

dir_ = args["images"]
orig = []
noisy = []
count = 0

for img_name in os.listdir(dir_):
    count += 1
    img = cv2.imread(dir_+img_name, 0)
    img = cv2.resize(img, (256, 80))
    noise = estimate_sigma(img)

    if "noisy" in img_name:
        noisy.append(noise)
    else:
        orig.append(noise)

print(max(orig), min(orig))
print(max(noisy), min(noisy))
print(np.mean(orig), np.mean(noisy))

pyplot.figure(1, (8,8))
ax = pyplot.subplot(1,2,1)
ax.hist(orig, bins=30)

ax = pyplot.subplot(1,2,2)
ax.hist(noisy, bins=40)

pyplot.show()