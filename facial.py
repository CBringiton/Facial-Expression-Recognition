# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 01:48:53 2018

@author: Chitranjan_Chhaba
"""

import pandas as pd
import numpy as np
dat=pd.read_csv('fer2013.csv')
a=dat.iloc[:,1].values
X_train=[]
for i in range(0,28709):
    X_train.append(np.asarray(a[i].split()))

x_train=np.array(X_train,dtype=float)
x_train=x_train.T
X_t=[]
for i in range(28709,35887):
    X_t.append(np.asarray(a[i].split()))

X_test=np.array(X_t, dtype = float).T
y_train=dat.iloc[:28709,0].values
Y_train=np.array(y_train)
y_test=dat.iloc[28709:35887,0].values
Y_test=np.array(y_test)
yot=np.zeros((28709,8))
for i in range(0,28709):
    yot[i][Y_train[i]]=1
yotest=np.zeros((7178,8))
for i in range(0,7178):
     yotest[i][Y_test[i]]=1
x_train=x_train/255
X_test=X_test/255

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
classifier.add(Dense(output_dim = 2304, init = 'uniform', activation = 'relu', input_dim = 2304))
classifier.add(Dense(output_dim = 2304, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1252, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1252, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train.T, yot, batch_size = 400, nb_epoch = 50)
y_pred = classifier.predict(X_test.T)


ans =[]
for i in range(0,len(y_pred)):
    v=0
    index=0
    for j in range(0,8):
        if(y_pred[i][j]>v):
            v=y_pred[i][j]
            index=j
    ans.append(index)

f=np.asarray(ans)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, f)
count=0        
for i in range(0,len(f)) :
    if(f[i]==y_test[i]):
        count+=1

print(count/len(f))      
    