#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #importing KNN class from scikit learn
from sklearn.metrics import accuracy_score

your_path ="E:\ML\PROJECTS\ Iris Flower Problem\dataset\iris.csv" # setting path to dataset

dataset = pd.read_csv(rf"{your_path}").values   # getting as numpy array
# print(dataset)
# labels
#'setosa', 'versicolor', 'virginica'  0 , 1 ,2 
# 150 data
# features
#Sepal-length Sepal-width Petal-length Petal-width 

data= dataset[:,0:4] #breaking in to data and target
target = dataset[:,-1]

#splitting data in to  train_data train_target  test_data test_target

train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.2) #20%
#KNN model
model = KNeighborsClassifier()

#training model
model.fit(train_data,train_target)

#testing  evaluation
predicted_target=model.predict(test_data)

#getting accuracy   num_of_predictions/total_predictions
acc = accuracy_score(predicted_target,test_target) #observation : accuracy >93 ==>so go for future prediction

#future prediction
a=model.predict([[ 6,3,4,4 ]]) # this function acept and return numpy array  [[],[],[],...]
print(a)

