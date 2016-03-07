#from __future__ import print_function
__author__ = 'Shuran'

from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2

import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
from sklearn import svm
from sklearn.externals import joblib



#initiate HOG descriptor

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#train_path = "train_model.pkl"
#classifier = joblib.load("train_model.pkl")
#print classifier.support_vectors_.reshape(-1,1)
#svm=cv2.SVM()
#svm.load(train_path)
#hog.setSVMDetector(classifier.support_vectors_.reshape(-1,1))
#print cv2.HOGDescriptor_getDefaultPeopleDetector()


#for imagepath in paths.list_images("test/test19.jpg"):

img = cv2.imread("test/test12.jpg")

img = imutils.resize(img, width=min(800,img.shape[1]))
orig = img.copy()

#find people in the image
(rects, weights) = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
print rects
print type(rects)
print rects.shape


#draw the bounding box around it
for (x,y,w,h) in rects:
    cv2.rectangle(orig,(x,y), (x+w,y+h), (0,0,255),2)

    #apply non-maxima suppression to the bounding boxes
#rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
#pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    #draw the final bounding boxes
#for(xA,yA,xB,yB) in pick:
 #   cv2.rectangle(orig,(xA,yA), (xB,yB), (0,0,255),2)

cv2.imshow('before', orig)
cv2.waitKey(0)




















