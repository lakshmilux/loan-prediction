#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:21 2018
@author: darling
"""
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler 
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import KernelPCA



#importing data
data = pd.read_csv("/home/darling/Documents/Data Science/Loan Prediction Project/train_data.csv")
data['Dependents'] = data['Dependents'].replace('3+','3')
values = data.values

#checking missing values
print(data.isnull().sum())



#imputing missing values for categorical variables
data['Gender'] = data['Gender'].fillna(data['Gender'].value_counts().index[0])
data['Married'] = data['Married'].fillna(data['Married'].value_counts().index[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].value_counts().index[0])

#imputing missing values for descrete variables
x = values[:,8:11]
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,0:3])
transformed_values = imputer.transform(x[:,0:3])
data['LoanAmount'] = transformed_values[:,0]
data['Loan_Amount_Term'] = transformed_values[:,1]
data['Credit_History'] = transformed_values[:,2]

y = values[:,3:4]
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(y[:,0:1])
transformed_values = imputer.transform(y[:,0:1])
data['Dependents'] = transformed_values[:,0]

#Encoding variables from object to category
data['Loan_ID'] = data['Loan_ID'].astype('category')
data['Gender'] = data['Gender'].astype('category')
data['Married'] = data['Married'].astype('category')
data['Education'] = data['Education'].astype('category')
data['Self_Employed'] = data['Self_Employed'].astype('category')
data['Property_Area'] = data['Property_Area'].astype('category')
data['Loan_Status'] = data['Loan_Status'].astype('category')

#converting variables to numeric
cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

#Splitting the data
d = data.values
x_train, x_test, y_train, y_test = train_test_split(d[:,0:12], d[:,12:], test_size = 0.25, random_state = 0)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#Applying PCA
pca = KernelPCA(n_components = 8, kernel='rbf')
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)
#explained_variance = pca.explained_variance_ratio_

#Model building

#Fitting model to KNN
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train, y_train)

#Fitting model to kernel SVM
svm_classifier = SVC(kernel = 'rbf', random_state = 0)
svm_classifier.fit(x_train, y_train)

#Fitting model to naive bayes
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

#Fitting model to Decision Tree
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(x_train, y_train)

#Fitting model to Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(x_train, y_train)

#Fitting model to xgboost
xg_classifier = XGBClassifier()
xg_classifier.fit(x_train, y_train)

#predicting the results
knn_pred = knn_classifier.predict(x_test)
svm_pred = svm_classifier.predict(x_test)
nb_pred = nb_classifier.predict(x_test)
dt_pred = dt_classifier.predict(x_test)
rf_pred = rf_classifier.predict(x_test)
xg_pred = xg_classifier.predict(x_test)

#validating the model with confusion matrics
knn_cm = confusion_matrix(y_test, knn_pred)     #125/128 correct predictions 29/26 incorrect predictions/ 78%/78% accuracy
svm_cm = confusion_matrix(y_test, svm_pred)     #129/128 correct predictions 25/26 incorrect predictions/ 80%/80% accuracy
nb_cm  = confusion_matrix(y_test, nb_pred)      #127/128 correct predictions 27/26 incorrect predictions/ 79%/79% accuracy
dt_cm = confusion_matrix(y_test, dt_pred)       #109/108 correct predictions 45/46 incorrect predictions/ 72%/71% accuracy
rf_cm = confusion_matrix(y_test, rf_pred)       #124/130 correct predictions 30/24 incorrect predictions/ 76%/76% accuracy
xg_cm = confusion_matrix(y_test, xg_pred)       #125/126 correct predictions 29/28 incorrect predictions/ 78%/78% accuracy

# evaluate an LDA model on the dataset using k-fold cross validation
accuracies = cross_val_score(estimator = xg_classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
