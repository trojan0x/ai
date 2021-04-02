# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:21:11 2020

@author: JEFF
"""

#import dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load dataset
data = load_breast_cancer()
#organize the data 
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

#split data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42)

#import and initialize our classifier 

gnb = GaussianNB()
model = gnb.fit(train, train_labels)
#Pridictions
prd = gnb.predict(test)
print(prd)

print(accuracy_score(test_labels, prd))

