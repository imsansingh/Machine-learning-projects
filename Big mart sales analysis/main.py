import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

mart_data = pd.read_csv('Train.csv')

mart_data.head()

mart_data.shape

mart_data.info()

correlation = mart_data.corr()

plt.figure()
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, cmap='Oranges' )
plt.show()

print(correlation['Item_Outlet_Sales'])

 #### Categorical features
     -> Item_Identifier
     -> Item_Fat_Content
     -> Item_Type
     -> Outlet_Identifier
     -> Outlet_Size
     -> Outlet_Location_Type
     -> Outlet_Type

mart_data.isnull().sum()

#### Handling missing values

# for Item_Weight -- we will find mean for Item_Weight and fill the missing values
m = mart_data['Item_Weight'].mean()
print(m)

mart_data['Item_Weight'].fillna(m, inplace=True)

mart_data.isnull().sum()

# for Outlet_Size -- as it has categorical data, we will find mode of Outlet_Size and fill the missing values
mode_of_outlet_size = mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

(mode_of_outlet_size)

missing_values = mart_data['Outlet_Size'].isnull()
print(missing_values)

mart_data.loc[missing_values, 'Outlet_Size'] = mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: 'Small')

mart_data.isnull().sum()

mart_data.describe()

# Item_Weight distribution
sns.set()
plt.figure()
sns.distplot(mart_data['Item_Weight'])
plt.show()

# Item_MRP distribution
sns.set()
plt.figure()
sns.distplot(mart_data['Item_MRP'])
plt.show()

# Item_Outlet_Sales distribution
sns.set()
plt.figure()
sns.distplot(mart_data['Item_Outlet_Sales'])
plt.show()

# Outlet_Establishment_Year count
sns.set()
plt.figure()
sns.countplot(x = 'Outlet_Establishment_Year', data=mart_data )
plt.show()

# Item_Type count
sns.set()
plt.figure(figsize=(20,8))
sns.countplot(x = 'Item_Type', data=mart_data )
plt.show()

# Outlet_Location_Type count
sns.set()
plt.figure(figsize=(4,4))
sns.countplot(x = 'Outlet_Location_Type', data=mart_data )
plt.show()

mart_data['Item_Fat_Content'].value_counts()

# here Low Fat, LF and low fat is same so we have to replace LF and low fat with Low Fat
mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

mart_data['Item_Fat_Content'].value_counts()

#### Label Encoding -- Changing categorical values to numerical values

encoder = LabelEncoder()

mart_data['Item_Identifier'] = encoder.fit_transform(mart_data['Item_Identifier'])

mart_data['Item_Fat_Content'] = encoder.fit_transform(mart_data['Item_Fat_Content'])

mart_data['Item_Type'] = encoder.fit_transform(mart_data['Item_Type'])

mart_data['Outlet_Identifier'] = encoder.fit_transform(mart_data['Outlet_Identifier'])

mart_data['Outlet_Size'] = encoder.fit_transform(mart_data['Outlet_Size'])

mart_data['Outlet_Location_Type'] = encoder.fit_transform(mart_data['Outlet_Location_Type'])

mart_data['Outlet_Type'] = encoder.fit_transform(mart_data['Outlet_Type'])

mart_data.head()

#### Splitting features and target

Xm = mart_data.drop(columns=['Item_Outlet_Sales','Item_Identifier'], axis=1)
Ym = mart_data['Item_Outlet_Sales']

Xm

Ym

#### Splitting data into training and testing data

Xm_train, Xm_test, Ym_train, Ym_test = train_test_split(Xm, Ym, test_size=0.2, random_state=2)

print(Xm.shape, Xm_train.shape, Xm_test.shape)

### Model Training
#### 1. Using XGBoost Regressor

model2_XGB = XGBRegressor()
model2_XGB.fit(Xm_train, Ym_train)

predicton_XGB = model2_XGB.predict(Xm_test)

# R squared error
error_score_XGB = metrics.r2_score(Ym_test, predicton_XGB)
print('R squared error is :', error_score_XGB)

Ym_test = list(Ym_test)
plt.scatter(Ym_test, predicton_XGB, color='red')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#### 2. Using Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

model2_RFR = RandomForestRegressor(n_estimators=100)
model2_RFR.fit(Xm_train, Ym_train)

predicton_RFR = model2_RFR.predict(Xm_test)
#print(predicton_RFR)

# R squared error
error_score_RFR = metrics.r2_score(Ym_test, predicton_RFR)
print('R squared error is :', error_score_RFR)

Ym_test = list(Ym_test)
plt.scatter(Ym_test, predicton_RFR, color='red')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#### 3. Using Linear Regression

from sklearn.linear_model import LinearRegression
model2_LR = LinearRegression()
model2_LR.fit(Xm_train, Ym_train)

predicton_LiR = model2_LR.predict(Xm_test)
print(predicton_RFR)

# R squared error
error_score_LiR = metrics.r2_score(Ym_test, predicton_LiR)
print('R squared error is :', error_score_LiR)

