import os
import cv2
import argparse
import numpy as np
from skimage import io, restoration, img_as_ubyte


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
ap.add_argument("-t", "--thresh", type=float, default=250.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

psf = np.ones((5,5)) / 25

sharpen_kernel_1 = np.array(
    [[0,-1,0],
    [-1,5,-1],
    [0,-1,0]]
)

sharpen_kernel_2 = np.array(
    [[-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]]
)

custom_kernel_1 = np.array(
    [[-1,-2,-1],
    [-2,13,-2],
    [-1,-2,-1]]
)

custom_kernel_2 = np.array(
    [[0,0,-1,0,0],
    [0,-2,-3,-2,0],
    [-1,-3,25,-3,-1],
    [0,-2,-3,-2,0],
    [0,0,-1,0,0]]
)

for img_name in os.listdir(args["images"]):
    img = io.imread(args["images"]+img_name, as_gray=True)
    img = cv2.resize(img, (256, 80))

    lap = cv2.Laplacian(img_as_ubyte(img), cv2.CV_64F).var()

    if lap < args["thresh"]:
        if lap < 100:
            iter_ = 40
        elif lap < 250:
            iter_ = 30
        else:
            iter_ = 25

        cv2.imshow("img is blured", img)

        img_sharpen_1 = cv2.filter2D(img, -1, sharpen_kernel_1)
        cv2.imshow("blur_img -> sharpen1", img_sharpen_1)

        img_sharpen_2 = cv2.filter2D(img, -1, sharpen_kernel_2)
        cv2.imshow("blur_img -> sharpen2", img_sharpen_2)

        img_custom_kernel_1 = cv2.filter2D(img, -1, custom_kernel_1)
        cv2.imshow("blur_img -> custom_kernel_1", img_custom_kernel_1)

        img_custom_kernel_2 = cv2.filter2D(img, -1, custom_kernel_2)
        cv2.imshow("blur_img -> custom_kernel_2", img_custom_kernel_2)

        img_richard = restoration.richardson_lucy(img, psf, iterations=iter_)
        cv2.imshow("blur_img -> richard", img_richard)

        row1 = np.hstack((img, img_sharpen_1, img_sharpen_2))
        row2 = np.hstack((img_custom_kernel_1, img_custom_kernel_2, img_richard))
        frame = np.vstack((row1, row2))
        cv2.imshow("img | sharpen1 | sharpen2 | custom1 | custom2 | richard", frame)

    else:
        cv2.imshow("img is not blured", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        break