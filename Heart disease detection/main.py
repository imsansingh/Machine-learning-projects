import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('heart_disease_data.csv')

data.head()

data.shape

data.size

data.info()

data.isnull().sum()

data.describe()

# checking the distribution of target value
data['target'].value_counts()

plt.figure()
sns.countplot(x='target', data=data)

plt.figure(figsize=(12,5))
sns.countplot(x='age', data=data)

# splitting features and target
X = data.drop(columns='target', axis=1)
Y = data['target']

X

Y

# splitting dataset into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y ,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

### Evaluating the model

from sklearn.metrics import accuracy_score, precision_score

# on training data
train_pred = model.predict(X_train)
train_data_accuracy = accuracy_score(Y_train, train_pred)
train_data_precision = precision_score(Y_train, train_pred)

print('Accuracy on training data :', train_data_accuracy)
print('Precison on training data :', train_data_precision)

# on testing data
test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, test_pred)
test_data_precision = precision_score(Y_test, test_pred)

print('Accuracy on training data :', test_data_accuracy)
print('Precison on training data :', test_data_precision)

### Building the predictive system

input_data = (20,1,0,140,187,0,0,144,1,4,2,2,3)

# changing the input data to numpy array
input_data_array = np.asarray(input_data)

# reshape the array as we are predicting for only one instance
input_data_reshape = input_data_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)
print(prediction)

if prediction[0]==1:
    print('!!Person has heart disease!!')
else:
    print('Person does not have heart disease')

