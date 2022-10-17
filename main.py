import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
"""
https://stackoverflow.com/questions/58751101/count-number-of-cells-in-the-image

image = cv2.imread("t1.png")
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hsv_lower = np.array([156,60,0])
hsv_upper = np.array([179,115,255])
mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

minimum_area = 200
average_cell_area = 650
connected_cell_area = 1000
cells = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > minimum_area:
        cv2.drawContours(original, [c], -1, (36,255,12), 2)
        if area > connected_cell_area:
            cells += math.ceil(area / average_cell_area)
        else:
            cells += 1
print('Cells: {}'.format(cells))
cv2.imshow('close', close)
cv2.imshow('original', original)
cv2.waitKey()
"""

# RGB
# [ 78, 79, 84 ]
# [ 74, 68, 70 ]

image = cv2.imread("c2.png")
# image = cv2.imread("t1.png")
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
# plt.imshow(hsv)
# plt.show()

# hsv_lower = np.array([156,60,0])
# hsv_upper = np.array([179,115,255])
hsv_lower = np.array([0, 0, 50])
hsv_upper = np.array([150, 120, 160])
mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# plt.imshow(close)
# plt.show()

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

minimum_area = 200
average_cell_area = 1000
connected_cell_area = 1000
cells = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > minimum_area:
        cv2.drawContours(original, [c], -1, (36, 255, 12), 2)
        # if area > connected_cell_area:
        #     cells += math.ceil(area / average_cell_area)
        # else:
            # cells += 1
        cells += 1
print('Cells: {}'.format(cells))
cv2.imshow('close', close)
cv2.imshow('original', original)
cv2.waitKey()
