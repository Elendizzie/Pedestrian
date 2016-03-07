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

(winW, winH) = (100,200)

hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

trainData = []
label = []

image1 = cv2.imread("pos/train1.jpg",0)

for(x,y,window) in help.window_slides(image1,stepSize=60,windowSize=(winW, winH)):
    if window.shape[0]!=winH or window.shape[1]!=winW:
        continue

    dps1 = hog.compute(window).flatten()

    trainData.append(dps1)
    label.append(1)

    #visualize all the process
    clone = image1.copy()
    cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0),2)
    cv2.imshow("window", clone)
    cv2.waitKey(1)
    time.sleep(0.025)


image2 = cv2.imread("neg/neg14.png",0)

for(x,y,window) in help.window_slides(image2,stepSize=60,windowSize=(winW, winH)):
    if window.shape[0]!=winH or window.shape[1]!=winW:
        continue


    dps2 = hog.compute(window).flatten()

    trainData.append(dps2)
    label.append(0)


    #visualize all the process
    clone = image2.copy()
    cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0),2)
    cv2.imshow("window", clone)
    cv2.waitKey(1)
    time.sleep(0.025)



new_trainData = np.array(trainData)
new_label = np.array(label)


print new_trainData.shape
print new_label.shape

#svm.train(new_trainData, new_label,params=svm_params)
#svm.save('train_data.xml')
#print new_trainData.shape

classifier.fit(new_trainData,new_label)
joblib.dump(classifier,'test_model.pkl')
score= classifier.predict_proba(dps1)
print type(score)
print score
print score[0][0]
print score[0][1]

#print type(new_label)