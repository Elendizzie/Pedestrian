__author__ = 'Shuran'
import cv2
import time
import help
from imutils import paths
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


svm_params = dict(kernal_type = cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=1, gamma=0.5)
classifier = svm.SVC(kernel='linear',probability= True, C=1)

(winW, winH) = (100,250)

#window size, block size, block stride, cell size, nbins
hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)


trainData = []
label = []
for imagepath in paths.list_images("pos"):
    image = cv2.imread(imagepath,0)

    for(x,y,window) in help.window_slides(image,stepSize=50,windowSize=(winW, winH)):
        if window.shape[0]!=winH or window.shape[1]!=winW:
            continue

        dps1 = hog.compute(window).flatten()
        print len(dps1)
        trainData.append(dps1)
        label.append(1)

        #visualize all the process
        clone = image.copy()
        cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0),2)
        cv2.imshow("window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)

for imagepath in paths.list_images("neg"):
    image = cv2.imread(imagepath,0)

    for(x,y,window) in help.window_slides(image,stepSize=50,windowSize=(winW, winH)):
        if window.shape[0]!=winH or window.shape[1]!=winW:
            continue


        dps1 = hog.compute(window).flatten()
        trainData.append(dps1)
        label.append(0)


        #visualize all the process
        clone = image.copy()
        cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0),2)
        cv2.imshow("window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)



new_trainData = np.array(trainData)
new_label = np.array(label)

#svm.train(new_trainData, new_label,params=svm_params)
#svm.save('train_data.xml')
#print new_trainData.shape
print new_trainData.shape
print new_label.shape

classifier.fit(new_trainData,new_label) #this is the training step
joblib.dump(classifier,'hard_train_model_0.09.pkl') #save the model

