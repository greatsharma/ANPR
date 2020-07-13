import os
import cv2
import time
import argparse
import numpy as np
from deskew import determine_skew
from tensorflow.keras.models import load_model
from skimage.restoration import estimate_sigma

from utils import rotate_image, tesseract_extract_text
from utils import modcrop, shave, image_enhancement


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
ap.add_argument("-nt", "--noise_thresh", type=float, default=1.2,
	help="noise level above  this value will be considered noisy")
ap.add_argument("-bt", "--blur_thresh", type=float, default=250.0,
	help="focus measures that fall below this value will be considered blurry")
ap.add_argument("-f", "--filter", type=str, default="gauss",
    help="allowed values are `gauss`, `median`")
ap.add_argument("-ocr", "--ocr", type=str, default="tesseract",
    help="allowed values are `tesseract`, `keras`")
ap.add_argument("-m", "--mode", type=str, default="prod",
    help="allowed values are `prod`, `debug`")
args = vars(ap.parse_args())


if args["ocr"] == "keras":
    import keras_ocr
    recognizer = keras_ocr.pipeline.Pipeline()

MODE = args["mode"]
denoiser = load_model("models/denoiser_ae2.h5")
psf = np.ones((5,5)) / 25
custom_kernel = np.array(
    [[0,0,-1,0,0],
    [0,-2,-3,-2,0],
    [-1,-3,25,-3,-1],
    [0,-2,-3,-2,0],
    [0,0,-1,0,0]]
)
morph_kernel = np.ones((3,3), np.uint8)


for img_name in os.listdir(args["images"]):
    print("\n\n")
    frame_name = ""
    tik = time.time()

    img_stage0 = cv2.imread(args["images"]+img_name)
    img_stage0 = cv2.resize(img_stage0, (256, 80))

    img_stage0 = image_enhancement(img_stage0)
    img_stage0 = cv2.cvtColor(img_stage0, cv2.COLOR_BGR2GRAY)
    img_stage0 = cv2.resize(img_stage0, (256, 80), cv2.INTER_CUBIC)

    frame = img_stage0.copy()
    frame_name += f" img_{img_name.split('_')[0]} "

    img_stage1 = img_stage0.copy()
    noise = estimate_sigma(img_stage1)

    if noise > args["noise_thresh"]:
        img_stage1 = img_stage1 / 255.
        img_stage1 = denoiser.predict(np.expand_dims(img_stage1, axis=(0,3)))[0, :, :, 0] * 255.
        img_stage1 = img_stage1.astype("uint8")
    else:
        if args["filter"] == "median":
            img_stage1 = cv2.medianBlur(img_stage1, 5)
        else:
            img_stage1 = cv2.GaussianBlur(img_stage1, (5,5),0)

    if MODE == "debug":
        frame = np.hstack((frame, img_stage1))
    frame_name += "-> denoised "

    img_stage2 = img_stage1.copy()
    lap = cv2.Laplacian(img_stage2, cv2.CV_64F).var()

    if lap < args["blur_thresh"]:
        img_stage2 = cv2.filter2D(img_stage2, -1, custom_kernel)
        if MODE == "debug":
            frame = np.hstack((frame, img_stage2))
        frame_name += "-> deblured "

    _, img_stage3 = cv2.threshold(img_stage2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if MODE == "debug":
        frame = np.hstack((frame, img_stage3))
    frame_name += "-> threshed "

    img_stage4 = cv2.morphologyEx(img_stage3, cv2.MORPH_CLOSE, morph_kernel) # first dilate than erode
    frame = np.hstack((frame, img_stage4))
    frame_name += "-> morphed "

    cv2.imshow(frame_name, frame)

    img_stage5 = img_stage4.copy()
    try:
        angle = determine_skew(img_stage4)
        if abs(angle) < 20:
            img_stage5 = rotate_image(img_stage4, angle, (0, 0, 0))
    except Exception as e:
        pass
    cv2.imshow("aligned img", img_stage5)

    if args["ocr"] == "keras":
        img_stage5 = np.dstack((img_stage5, img_stage5, img_stage5))
        predictions = recognizer.recognize(img_stage5)
        text = ""
        for pred in predictions[0]:
            text += pred[0]
    else:
        text = tesseract_extract_text(img_stage5, min_conf=0, mode=MODE)
        
    print("\ntext: ", text)
    print("time taken: ", round(time.time()-tik, 5))

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()