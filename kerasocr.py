import os
import cv2
import time
import argparse
import keras_ocr
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())

dir_ = args["images"]
recognizer = keras_ocr.pipeline.Pipeline()
# avg time per image: 1.23 sec

for img_name in os.listdir(dir_):
    img = cv2.imread(dir_+img_name)
    img = cv2.resize(img, (256,80))
    
    tik = time.time()
    predictions = recognizer.recognize(img) # (text, box)
    print("\n\n tt: ", time.time() - tik)

    text = ""
    for pred in predictions[0]:
        text += pred[0]
    print(text)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()