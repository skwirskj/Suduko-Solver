import cv2
import numpy as np

image = cv2.imread('sudukoBoard1.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', grayImage)
cv2.waitKey(0)
edges = cv2.Canny(grayImage, 50, 150, apertureSize=3)
cv2.imshow('edge', edges)
cv2.waitKey(0)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('sudukoBoard', image)
cv2.waitKey(0)


