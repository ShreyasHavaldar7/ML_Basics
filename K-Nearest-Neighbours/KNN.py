#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:40:57 2019

@author: shreyas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import operator
import itertools
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

dataset = pd.read_csv("iris.csv")

train, test = train_test_split(dataset, test_size = 0.4)

def dist(d1, d2, len):
    d=0
    for i in range(len):
        d = d+ math.sqrt(math.pow((d1[i]-d2[i]), 2))
        
    return d



k =5

ans =[]

for i in range(len(test)):
    distance = []
    length = len(test.iloc[i]) - 1
    
    for j in range(len(train)):
        distance.append((train.iloc[j], dist(train.iloc[j], test.iloc[i], length)))
    
    distance.sort(key = operator.itemgetter(1))
    
    neighbours = []
    for j in range(k):
        neighbours.append(distance[j][0])
        
    votes = {}
    
    for j in range(len(neighbours)):
        vote = neighbours[j][-1]
        if vote in votes:
            votes[vote]= votes[vote] +1
        else:
            votes[vote]=1
            
    maxVotes = sorted(votes.items(), key = operator.itemgetter(1), reverse = True)
    
    ans.append(maxVotes[0][0])
    
target = test.iloc[:, 4].values
target = target.reshape(len(target),1)

cf = confusion_matrix(target, ans)

print('Confusion Matrix is: \n', cf)

accuracy = (cf[0][0]+ cf[1][1] + cf [2][2])/len(test)
print('Accuracy is: ', accuracy)

#correct = 0
#for i in range(len(test)) :
#    
#    if test.iloc[i][-1] == ans[i]:
#        
#        correct = correct +1
#    
#print((correct / len(test)) * 100)    
