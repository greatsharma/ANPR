import os
import cv2
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image directory")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory, where the noisy images are stored")
ap.add_argument("-ic", "--initial_count", required=True,
	help="initial image count")
args = vars(ap.parse_args())

rng = np.random.RandomState(42)

def add_noise(X, noise=0.08):
    X += noise * rng.normal(loc=0.0, scale=1.0, size=X.shape)
    X = np.clip(X, 0., 1.)
    return X


if not "/" == args["input"][-1]:
    args["input"] += "/"
if not "/" == args["output"][-1]:
    args["output"] += "/"


IMG_SHAPE = (256, 80)

count = int(args["initial_count"])
for img_name in os.listdir(args["input"]):
    try:
        img = cv2.imread(args["input"]+img_name, 0)
        cv2.imwrite(args["output"] + f"{count}_orig.png", cv2.resize(img, IMG_SHAPE))
        
        img = img.astype("float64")
        img /= 255.
        img = add_noise(img)
        img *= 255
        img = img.astype("uint8")
        img = cv2.resize(img, IMG_SHAPE)
        
        cv2.imwrite(args["output"] + f"{count}_noisy.png", img)
        count += 1
    except Exception as e:
        print(img_name)
        print(e)
        os.remove(args["output"] + f"orig_{count}.png")

print(count)