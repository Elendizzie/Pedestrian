__author__ = 'Shuran'
#from __future__ import print_function


from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import numpy as np
import imutils
import time
import help
from sklearn import svm
from sklearn.externals import joblib
import pickle

#initiate HOG descriptor

hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#train_path = "train_data.xml"
#svm=cv2.SVM()
#svm.load(train_path)
#hog.setSVMDetector(train_path)
(winW, winH) = (100,250)
#classifier = svm.SVC(kernel='linear',probability= True)

classifier = joblib.load("hard_train_model.pkl")

rects = []
#for imagepath in paths.list_images("test"):
image = cv2.imread("pos/pos48.png")

for resized in help.pyramid(image, scale=1.5):

    for(x,y,window) in help.window_slides(resized,stepSize=60,windowSize=(winW, winH)):
        if window.shape[0]!=winH or window.shape[1]!=winW:
            continue

        dps2 = hog.compute(window).flatten()
        verify_label = classifier.predict_proba(dps2)
        prob_pedestrian = verify_label[0][1]
        print verify_label
        if prob_pedestrian>0.6:
            point=[x,y,winW,winH]
            rects.append(point)
            print point
    #visualize all the process
        clone = resized.copy()
        cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0),2)
        cv2.imshow("window", clone)
        cv2.waitKey(1)
        time.sleep(0.02)

    '''   cv2.rectangle(resized, (x,y), (x+winW, y+winH), (0,0,255),2)
            print "1"
        else:
            print "0"'''
#rects.append([40,30,295,590])

rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.5)
for(xA,yA,xB,yB) in pick:
    cv2.rectangle(image,(xA,yA), (xB,yB), (0,0,255),2)

cv2.imshow('after', image)
cv2.waitKey(0)

print rects
print type(rects)
print rects.shape

#the first prob stands for the probability of not a pedestrian, and the second prob stands for the probability of a pedestrian