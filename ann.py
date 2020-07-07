# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:11:43 2020

@author: Vincent
"""

# Kaggle Titanic prediction


# Basic ANN

import pandas as pd
import numpy as np

## Loading data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

y_train = train_data.Survived
X_train = train_data.iloc[:,2:]
X_test = test_data.iloc[:,1:]

X_train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
X_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
X_all = pd.concat([X_train,X_test],ignore_index=True)


## Missing data imputation
from sklearn.impute import SimpleImputer

imp_age = SimpleImputer(missing_values=np.nan, strategy='median')
imp_age.fit(X_all.Age.values.reshape(-1,1))
X_train.Age = imp_age.transform(X_train.Age.values.reshape(-1,1))
X_test.Age = imp_age.transform(X_test.Age.values.reshape(-1,1))

imp_embarked = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_embarked.fit(X_all.Embarked.values.reshape(-1,1))
X_train.Embarked = imp_embarked.transform(X_train.Embarked.values.reshape(-1,1))
X_test.Embarked = imp_embarked.transform(X_test.Embarked.values.reshape(-1,1))

imp_fare = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_fare.fit(X_all.Fare.values.reshape(-1,1))
X_train.Fare = imp_fare.transform(X_train.Fare.values.reshape(-1,1))
X_test.Fare = imp_age.transform(X_test.Fare.values.reshape(-1,1))


# Encode categorical data and scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Pclass', 'Sex', 'Embarked']),
        (StandardScaler(), ['Age', 'SibSp', 'Parch', 'Fare'])
        )
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.fit_transform(X_test)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
final_predictions = (y_pred > 0.5)

# Output prediction file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': 1*final_predictions[:,0]})
output.to_csv('vincent_submission_ann.csv', index=False)