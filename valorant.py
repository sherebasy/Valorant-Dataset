# import the necessary packages
import numpy as np
import pytesseract
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# OCR the input image using Tesseract
options = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
text = pytesseract.image_to_string(gray, config=options)
print(text)
# show the final output image
cv2.imshow("Final", gray)
cv2.waitKey(0)

