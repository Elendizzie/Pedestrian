__author__ = 'Shuran'
import cv2
import time
import help
from imutils import paths
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

svm_params = dict(kernal_type = cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=1, gamma=0.5)
classifier = svm.SVC(kernel='linear',probability= True)

(winW, winH) = (100,250)

hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

trainData = []
label = []


