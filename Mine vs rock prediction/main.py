## Classification

### Mine vs Rock Prediction

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore')

data1 = pd.read_csv('sonar_data.csv', header=None)

data1.head(10)

data1[60]=data1[60].map({'R':0,'M':1})

data1.shape

data1.isnull()

data1.isnull().sum()

data1.describe()

data1[60].value_counts()

d=data1.groupby(60)
d.mean
# mean value of each column for M(mine) and R(rock)

# separating the data and label
X = data1.drop(columns=60, axis=1)
Y = data1[60] # target variable

X

Y

### Splitting the data into train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

X_train

print(X.shape, X_train.shape, X_test.shape)

### Model trainning :
####  1. Using KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score


model1_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model1_KNN.fit(X_train, Y_train)

# checking accuracy on training data
X_train_prediction = model1_KNN.predict(X_train)
traning_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on the training data : ',traning_data_accuracy)

X_train_prediction

# checking accuracy on testing data
X_test_prediction = model1_KNN.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,X_test_prediction)

print('Accuracy on the test data : ',test_data_accuracy)

test_data_precision = precision_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_precision)

test_data_f1 = f1_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_f1)

#### 2. Using Logistic Regression 

from sklearn.linear_model import LogisticRegression

model1_LR = LogisticRegression()

# training the model with training data
model1_LR.fit(X_train, Y_train)

# checking accuracy on training data
X_train_prediction = model1_LR.predict(X_train)
traning_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on the training data : ',traning_data_accuracy)

# checking accuracy on testing data
X_test_prediction = model1_LR.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on the testing data : ',testing_data_accuracy)

test_data_precision = precision_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_precision)

test_data_f1 = f1_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_f1)

#### 3. Using Support Vector Machine

from sklearn import svm
model1_SVM = svm.SVC(kernel='linear')

# training the model with training data
model1_SVM.fit(X_train, Y_train)

# checking accuracy on testing data
X_test_prediction = model1_SVM.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy using Support Vector Machine : ', testing_data_accuracy)

test_data_precision = precision_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_precision)

test_data_f1 = f1_score( Y_test,X_test_prediction)
print('f1 score on the test data : ',test_data_f1)

#### 4. Using Decision tree

from sklearn.tree import DecisionTreeClassifier

model1_DT = DecisionTreeClassifier()
model1_DT.fit(X_train, Y_train)

X_test_prediction = model1_DT.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on the test data : ',test_data_accuracy)

test_data_precision = precision_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_precision)

test_data_f1 = f1_score( Y_test,X_test_prediction)
print('f1 score on the test data : ',test_data_f1)

#### 5. Using Random forest

from sklearn.ensemble import RandomForestClassifier

model1_RF = RandomForestClassifier(n_estimators=5, criterion='entropy')
model1_RF.fit(X_train, Y_train)

X_test_prediction = model1_RF.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on the test data : ',test_data_accuracy)

test_data_precision = precision_score( Y_test,X_test_prediction)
print('Precision on the test data : ',test_data_precision)

test_data_f1 = f1_score( Y_test,X_test_prediction)
print('f1 score on the test data : ',test_data_f1)

### Making a predictive system

# Accuracy was highest when we used KNN classifier so we will use that model for our predictive system

input1 = (0.0164,0.0173,0.0347,0.0070,0.0187,0.0671,0.1056,0.0697,0.0962,0.0251,0.0801,0.1056,0.1266,0.0890,0.0198,0.1133,0.2826,0.3234,0.3238,0.4333,0.6068,0.7652,0.9203,0.9719,0.9207,0.7545,0.8289,0.8907,0.7309,0.6896,0.5829,0.4935,0.3101,0.0306,0.0244,0.1108,0.1594,0.1371,0.0696,0.0452,0.0620,0.1421,0.1597,0.1384,0.0372,0.0688,0.0867,0.0513,0.0092,0.0198,0.0118,0.0090,0.0223,0.0179,0.0084,0.0068,0.0032,0.0035,0.0056,0.0040)
# converting input data into numpy array
input1_array = np.asarray(input1)

reshaped_input1 = input1_array.reshape(1,60)
#print(reshaped_input1)

prediction1 = model1_KNN.predict(reshaped_input1)

print(prediction1)

if(prediction1[0]==0):
    print('The object is a Rock')
else:
    print('The object is a Mine')





















