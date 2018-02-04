# -*- coding: utf-8 -*-
"""
Created on Sun Feb 04 17:47:07 2018

@author: SKT
"""

import imutils
from imutils import face_utils
import dlib
import cv2
import math

def midPoint(x1, y1, x2, y2):
    return ((x2+x1)/2, (y2+y1)/2)

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1) 

features = {'jaw' : (0, 17),
            'right_eyebrow' : (17, 22),
            'left_eyebrow' : (22, 27),
            'nose' : (27, 36),
            'right_eye' : (36, 42),
            'left_eye' : (42, 48),
            'mouth' : (48, 68)}

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('images/example_02.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):    
    
    print('\nface' + str(i+1))
    
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    jaw = features['jaw']
    pts = shape[jaw[0]:jaw[1]]
    
    #end points of jaw
    x1, y1, x2, y2 = pts[0, 0], pts[0, 1], pts[-1, 0], pts[-1, 1]
    
    #calculate jaw length and draw a line
    jawLength = dist(x1, y1, x2, y2)
    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)
    
    print('jaw length: ' + str(jawLength))
    
    right_eye = features['right_eye']
    pts = shape[right_eye[0]:right_eye[1]]
    
    #finding corners of right eye
    x1, y1, x2, y2 = pts[0, 0], pts[0, 1], pts[3, 0], pts[3, 1]
    
    #midpoint of right eye
    r_midx, r_midy = midPoint(x1, y1, x2, y2)
    
    left_eye = features['left_eye']
    pts = shape[left_eye[0]:left_eye[1]]
    
    #finding corners of left eye
    x1, y1, x2, y2 = pts[0, 0], pts[0, 1], pts[3, 0], pts[3, 1]
    
    #midpoint of left eye
    l_midx, l_midy = midPoint(x1, y1, x2, y2)
    
    #calculate eye distance and draw a line
    eyeDist = dist(l_midx, l_midy, r_midx, r_midy)
    cv2.line(image, (l_midx, l_midy), (r_midx, r_midy), (255,0,0), 1)
    
    print('eye distance: ' + str(eyeDist))
       
    print('ratio: ' + str(eyeDist/jawLength))
    
cv2.imshow("Image", image)       
cv2.waitKey(0)