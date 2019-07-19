#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:50:48 2019

@author: shreyas
"""


from numpy import array
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Reading Data and Storing in matrices
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Reshaping Input Matrix to include a column of ones to account for the constant term
X = X.reshape((len(X), 1))
X0 = np.ones((30,1))
X = np.hstack((X0,X))

#Reshaping Output Matrix to with a column of ones to account for constant term
y = y.reshape((len(y), 1))
y0 = np.ones((30,1))
y = np.hstack((y0,y))

#to select 20 random points to be trained upon
train_indexes = random.sample(range(0,30),20)

#creating training input and output
X_train = X[train_indexes]
y_train = y[train_indexes]

# Making the test set
test_indexes = [index for index in range(indices) if index not in train_indexes]
X_test = X[test_indexes]
y_test = y[test_indexes]


#Calculating the moore-penrose inverse to get the value of weights in the form of matrix b
b = inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)

#predicting outputs by multiplying the input data matrix by the weight matrix
ytrainhat = X_train.dot(b)
ytesthat = X_test.dot(b)


# Visualising the Training set results
plt.scatter(X_train[:,1], y_train[:,1], color = 'red')
plt.plot(X_train[:,1], ytrainhat[:,1], color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test[:,1], y_test[:,1], color = 'red')
plt.plot(X_test[:,1], ytesthat[:,1], color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
