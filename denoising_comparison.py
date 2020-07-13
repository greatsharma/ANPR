import os
import cv2
import time
import argparse
import numpy as np
from scipy import fftpack
from tensorflow.keras.models import load_model
from skimage.restoration import estimate_sigma


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())

denoiser = load_model("models/denoiser_ae2.h5")
dir_ = args["images"]

for img_name in os.listdir(dir_):
    print("\n\n", img_name)

    img = cv2.imread(dir_+img_name, 0)
    img = cv2.resize(img, (256, 80))
    noise = estimate_sigma(img)
    print("noise: ", noise)

    if noise > 7:
        img_denoised_cv = img.copy()
        tik = time.time()
        cv2.fastNlMeansDenoising(img, img_denoised_cv, h=30, templateWindowSize=5, searchWindowSize=19)
        print("cv time: ", time.time() - tik)
        img_denoised_cv = img_denoised_cv / 255.

        img = img / 255.

        tik = time.time()
        img_fft = fftpack.fft2(img)
        keep_frac = 0.15
        img_fft2 = img_fft.copy()
        r, c = img_fft2.shape
        img_fft2[int(r*keep_frac):int(r*(1-keep_frac))] = 0
        img_fft2[:, int(c*keep_frac):int(c*(1-keep_frac))] = 0
        img_denoised_fft = fftpack.ifft2(img_fft2).real
        print("fft time: ", time.time() - tik)

        tik = time.time()
        img_denoised_ae = denoiser.predict(np.expand_dims(img, axis=(0,3)))[0, :, :, 0]
        print("ae time: ", time.time() - tik)

        if "noisy" in img_name:
            img_name = img_name.split("_")[0] + "_orig.png"
            img_orig = cv2.imread(dir_+img_name, 0)
            img_orig = cv2.resize(img_orig, (256, 80)) / 255
            frame = np.hstack((img, img_denoised_fft, img_denoised_cv, img_denoised_ae, img_orig))
            cv2.imshow("img_noisy | img_denoised_fft | img_denoised_CV | img_denoised_AE  | img_orig", frame)
        else:
            frame = np.hstack((img, img_denoised_fft, img_denoised_cv, img_denoised_ae))
            cv2.imshow("img_orig | img_denoised_fft | img_denoised_CV | img_denoised_AE", frame)
    else:
        cv2.imshow("img has no noise", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        break