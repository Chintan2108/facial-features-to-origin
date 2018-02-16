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
import os
from save_data import *

def midPoint(x1, y1, x2, y2):
    return ((x2+x1)/2, (y2+y1)/2)

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1) 

def eyeDist():
    
    global shape, features, image
    
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
    eye_dist = dist(l_midx, l_midy, r_midx, r_midy)
    cv2.line(image, (l_midx, l_midy), (r_midx, r_midy), (255,0,0), 1)
    
    #print('eye dist: ' + str(eye_dist))
    
    return eye_dist

def jawLength():
    
    global shape, features, image
    
    jaw = features['jaw']
    pts = shape[jaw[0]:jaw[1]]
    
    #end points of jaw
    x1, y1, x2, y2 = pts[0, 0], pts[0, 1], pts[-1, 0], pts[-1, 1]
    
    #calculate jaw length and draw a line
    jaw_length = dist(x1, y1, x2, y2)
    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)
    
    #print('jaw length: ' + str(jaw_length))
    
    return jaw_length

def nasalLength():
    
    global shape, features, image
    
    #finding nasal length
    nose = features['nose']
    pts = shape[nose[0]:nose[1]]
    x1, y1 = pts[0, 0], pts[0, 1]
    jaw = features['jaw']
    pts = shape[jaw[0]:jaw[1]]
    x2, y2 = pts[8, 0], pts[8, 1]
    nasal_length = dist(x1, y1, x2, y2)
    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1) 

    #print('nasal length: ' + str(nasal_length))
    
    return nasal_length

def mouthLength():
    
    global shape, features, image
    
    #finding mouth width
    mouth = features['mouth']
    pts = shape[mouth[0]:mouth[1]]
    
    x1, y1, x2, y2 = pts[0, 0], pts[0, 1], pts[6, 0], pts[6, 1]
    
    mouth_length = dist(x1, y1, x2, y2)
    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)
    
    #print('mouth length: ' + str(mouth_length))
    
    return mouth_length
    

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

genders = ['male', 'female']
geo_locs = ['african', 'chinese'] 

limit = 20

for geo_loc in geo_locs:
    for gender in genders:
        
        dir_path = 'images/' + geo_loc +'/' + gender + '/' 
        
        count = 1
        for filename in os.listdir(dir_path):
            
            if count>limit:
                break
            
            file_path = dir_path + '/' + filename
        
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(file_path)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # detect faces in the grayscale image
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):    
                
                #print('\nface' + str(i+1))
                
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
            
                eye_dist = eyeDist()
                
                jaw_length = jawLength()
                jaw_ratio = jaw_length/eye_dist
                #print('jaw ratio: ' +  str(jaw_ratio))
                
                nasal_length = nasalLength()
                nasal_ratio = nasal_length/eye_dist
                #print('nasal ratio: ' + str(nasal_ratio))
                
                mouth_length = mouthLength()
                mouth_ratio = mouth_length/eye_dist
                #print('mouth ratio: ' + str(mouth_ratio))
                
                addRow(geo_loc, gender, jaw_ratio, nasal_ratio, mouth_ratio)
                count += 1
                
                cv2.imwrite('dataset/' + geo_loc + '_' + gender + '_' + str(count) + '.jpg' , image)
                
            #cv2.imshow("Image", image)       
            #cv2.waitKey(0)
saveDF()           