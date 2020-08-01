import cv2
from matplotlib import pyplot as plt
import numpy as np
import operator


def preprocess_board(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Applying gaussian blur
    processed_img = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    # Adaptive threshold with 11 nearest neighbor pixels
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Inverting the colors to have non-zero pixel values along the gridlines of the puzzle.
    processed_img = cv2.bitwise_not(processed_img, processed_img)

    return processed_img


def find_corners(img):
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    shape = contours[0]

    # Getting the index of the corner points
    btm_right_pt, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in shape]), key=operator.itemgetter(1))
    top_left_pt, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in shape]), key=operator.itemgetter(1))
    btm_left_pt, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in shape]), key=operator.itemgetter(1))
    top_right_pt, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in shape]), key=operator.itemgetter(1))

    return [shape[top_left_pt][0], shape[top_right_pt][0], shape[btm_right_pt][0], shape[btm_left_pt][0]]


img_path = "sudukoBoard1.png"
proc_board = preprocess_board(img_path)
corners = find_corners(proc_board)
print(corners)
cv2.imshow('sudukoBoard', proc_board)
cv2.waitKey(0)



