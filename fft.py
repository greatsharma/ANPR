import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())

dir_ = args["images"]
for img_name in os.listdir(dir_):
    img = cv2.imread(dir_+img_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("error while reading image")

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2
    winsize = 6 // 2
    fshift[crow-winsize: crow+winsize, ccol-winsize: ccol+winsize] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.figure(1, (8,2))
    plt.subplot(1,3,1), plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2), plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([]) # HPF: high pass filter
    plt.show()