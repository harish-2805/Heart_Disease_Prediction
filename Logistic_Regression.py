import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Dataset/heart.csv')
X = dataset.iloc[: , : -1].values
Y = dataset.iloc[: , -1].values

# encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 6, 8 ,10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

##splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(X,Y , test_size = 0.2 , random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# training the logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

#predicting the test set results
Y_pred = classifier.predict(X_test)

#makign the confusion matrix for evaluating the performance of our model
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(Y_test , Y_pred)
accuracy = accuracy_score(Y_test , Y_pred)

#evaluating the performance of the model using stratified k-fold cross validation

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
skf = StratifiedKFold(n_splits = 10 , random_state = 0, shuffle = True)

accuracies = cross_val_score( estimator = classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean = accuracies.mean()
variance = accuracies.std()


print(f"Cross Validation Accuracy: {mean*100:.2f} %")
print(f"Standard Deviation: {variance*100:.2f} %")