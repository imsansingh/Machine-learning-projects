import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

cc_data = pd.read_csv('creditcard.csv')

cc_data.head()

cc_data.shape

cc_data.size

cc_data.info()

cc_data.describe()

cc_data.isnull().sum()

# distribution of legit and fraud transaction 
cc_data['Class'].value_counts()

#### This data is highly unbalanced as number of legit transactions(0) is very very greater than number of fraud transactions
##### 0 -> Normal transaction
##### 1 -> Fraud transaction

# separating the data for analysis
legit = cc_data[cc_data.Class==0]
fraud = cc_data[cc_data.Class==1]

print(legit.shape, fraud.shape)

fraud.head()

# Statistical measures of 'Amount' in legit
legit.Amount.describe()

fraud.Amount.describe()

# compare the values for both transactions
cc_data.groupby('Class').mean()

### Under Sampling
##### Build a sample dataset(because origional dataset is highly unbalanced) containing similar distribution of both transactions

legit_sample = legit.sample(n=500)

#### Concatinating two dataframes

new_ds = pd.concat([legit_sample, fraud], axis=0) # concatinating df row wise(axis=0)

new_ds.head()

new_ds['Class'].value_counts()

new_ds.groupby('Class').mean()

### Splitting the data into features and target

X = new_ds.drop(columns=['Class'], axis=1)
Y = new_ds['Class']

X

correalation = X.corr()

plt.figure(figsize=(20,20))
sns.heatmap(correalation, annot=True, cbar=False)
plt.show()

#X = X.drop(columns=['V1','V5','V16','V18'])

Y

### Splitting the data training and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

### Model training
#### 1.Using Logistic Regression

model_LR = LogisticRegression()

model_LR.fit(X_train, Y_train)

### Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# accuracy on training data
X_train_prediction = model_LR.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data :', training_data_accuracy)

training_data_precision = precision_score(Y_train, X_train_prediction)
print('Precision on training data :', training_data_precision)

# accuracy on testing data
X_test_prediction = model_LR.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on training data :', test_data_accuracy)

test_data_precision = precision_score(Y_test, X_test_prediction)
print('Precision on training data :', test_data_precision)

### Making a predictive system

input = (406,-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919,0)

# converting input data into numpy array as arrays are fast
input_array = np.asarray(input)

# reshaping the input data as we are predicting for 1 instance
reshaped_input = input_array.reshape(1,-1)

prediction = model_LR.predict(reshaped_input)
print(prediction)

if(prediction[0]==0):
    print('Legit transaction')
else:
    print('Fraud transaction')
