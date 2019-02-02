# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:41:55 2019

@author: Nomula Dheeraj Kumar
"""
'''
Dataset Link :  https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
Dataset consists of 48x48 pixel grayscale images of faces.

'''

#IMPORTING THE NECCESSARY LIBRARIES

import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras import optimizers
from keras.utils import np_utils

#OUTPUT THIS MODEL
output={0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

#IMPORTING THE DATASET
dataset=pd.read_csv('D:\Study\project\\fer2013.csv')

#COLLECTING TRAINING AND TEST DATASETS
Train_Data=dataset[(dataset['Usage']=='Training')]
Test_Data=dataset[dataset['Usage']=='PublicTest']

#SEGREGATING DEPENDENT AND INDEPENDENT VARIABLES
X_train=Train_Data['pixels'].values
y_train=Train_Data['emotion'].values
X_test=Test_Data['pixels'].values
y_test=Test_Data['emotion'].values

#CONVERTING INTO ndArray
L=[]
for i in range(len(X_train)):
    L.append(X_train[i].split(' '))
X_train = np.asarray(L, dtype=np.int64)

L=[]
for i in range(len(X_test)):
    L.append(X_test[i].split(' '))
X_test=np.asarray(L,dtype=np.int64)


#RESHAPING EACH ROW OF SIZE 2304 INTO 2D ARRAY OF SIZE 48X48 AND INCLUDING 1 TO INDICATE IT IS GRAY SCALE IMAGE
X_train=X_train.reshape(len(X_train),48,48,1)
X_test=X_test.reshape(len(X_test),48,48,1)

#SCALING TRAINING AND TEST DATA
X_train=X_train/255
X_test=X_test/255

#ENCODING THE CATEGORICAL DATA
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#BUILDING THE CNN 

#INITIALISING CNN
classifier=Sequential()

#ADDING CONV,POOLING LAYERS AND INTRODUCING NECESSARY DROPOUTS AT EACH LAYER
classifier.add(Conv2D(32,(3,3),input_shape=(48,48,1),activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(128,(3,3),activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(256,(3,3),activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#FLATTENING
classifier.add(Flatten())

#ADDING FULLY CONNECTED NEURAL NETWORK
classifier.add(Dense(units=128,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
classifier.add(Dropout(0.2))

#ADDING THE OUTPUT LAYER WHICH CONSISTS OF 7 NODES
classifier.add(Dense(units=7,activation='softmax'))

#CHOOSING THE OPTIMIZER AND LEARNING RATE
optimizer_choosen=optimizers.Adam(lr=0.00001)
classifier.compile(optimizer=optimizer_choosen,loss='categorical_crossentropy',metrics=['accuracy'])

#FITTING THE DATASET 
classifier.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200)

#SAVING THE MODEL
classifier.save('D:\Study\project\FACIAL_EXPRESSION.h5')


#REAL TIME FACIAL EXPRESSION DETECTION USING OPENCV

#LOADING THE MODEL
from keras.models import load_model
classifier=load_model('D:\Study\project\FACIAL_EXPRESSION.h5')

video=cv2.VideoCapture(0)
count=0
font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade=cv2.CascadeClassifier('D:\Study\project\haarcascade_frontalface_default.xml')

while True:
    count=count+1
    
    #CAPTURING THE IMAGE
    check,img=video.read()
    
    #RETURNS TRUE IF FRAME HAS BEEN GRABBED
    print(check)
    if(check==True):
        
        #CONVERTING INTO GRAYSCALE IMAGE
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        
        #DETECTING VARIOUS FACES IN THE FRAME
        faces=faceCascade.detectMultiScale(gray,scaleFactor=1.1)
        
        #DRAWING RECTANGLE AROUND THE FACES
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,127,0),2,5)
            
            #CROPPING THE IMAGE
            face_crop=img[y:y+h,x:x+w]
            
            #ADJUSTING THE CROPPED IMAGE TO OUR REQUIRED INPUT SIZE
            face_crop=cv2.resize(face_crop,(48,48))
            face_crop=cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
            (size_x,size_y)=face_crop.shape
            face_crop=face_crop.reshape(1,size_x,size_y,1)
            
            #PREDICTING THE FACIAL EXPRESSION
            prediction=classifier.predict(face_crop)
            
            #FINDING THE BEST MATCH
            prediction=np.argmax(prediction)
            
            #GGETTING CORRESPONDING EXPRESSION
            your_expression=output[prediction]
            cv2.putText(img,your_expression,(x,y),font,1,(200,0,0),3,cv2.LINE_AA)
        
        #DISPLAYING THE IMAGE
        cv2.imshow('Let\'s See your expression:',img)
        
        #TO STOP CPATURING WHEN THE KEY 'Q' IS PRESSED
        key=cv2.waitKey(1)
        if key==ord('q'): #Press q to exit 
            break

#NO OF IMAGES CAPTURED
print(count)

#CLOSING THE CAPTURING DEVICE
video.release()

cv2.destroyAllWindows()
