import os
import cv2
import argparse
import numpy as np
import pytesseract
from pytesseract import Output


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to image directory")
args = vars(ap.parse_args())


dir_ = args["images"]
for img_name in os.listdir(dir_):
    img = cv2.imread(dir_+img_name)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pytesseract.image_to_data(img, output_type=Output.DICT)
    # print(results)

    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]
        conf = int(results["conf"][i])

        if conf > 0:
            print(f"Confidence: {conf}")
            print(f"Text: {text}")
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out_frame = np.hstack((img_orig, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    cv2.imshow("out", out_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()