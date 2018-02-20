# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:42:22 2018

@author: Chintan
"""

import numpy as np
import cv2

min_Y = np.array([0,133,77],np.uint8)
max_Y = np.array([255,173,127],np.uint8)
img = cv2.imread('images/simple-face.jpg')
imgY = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
skinRegion = cv2.inRange(imgY, min_Y, max_Y)

#identifying contours
_, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#plot contours
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(img, contours, i, (0,255,0), 3)

color = 'Null'

print('dominant color : %s' % color)
#display image
cv2.imshow("image1",img)
cv2.waitKey(0)


