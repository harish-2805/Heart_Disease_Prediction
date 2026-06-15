import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the required libraries for evaluating the performance of the model

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score
)

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
X_test = sc.transform(X_test)


# NOW WE WILL IMPLEMENT ALL THE CLASSIFICATION ALGORITHMS AND EVALUATE THEIR PERFORMANCE USING STRATIFIED K-FOLD CROSS VALIDATION

# 1. LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
Log_classifier = LogisticRegression(random_state = 0)
Log_classifier.fit(X_train,Y_train)

#predicting the test set results
Y_pred_log = Log_classifier.predict(X_test)

#evaluating the performance of the model using confusion matrix and classification report

print("\n================ LOGISTIC REGRESSION ================")

cm_log = confusion_matrix(Y_test , Y_pred_log)
print("\nConfusion Matrix:")
print(cm_log)

print("\nClassification Report:")
print(classification_report(Y_test , Y_pred_log))

roc_log = roc_auc_score(Y_test , Y_pred_log)
print(f"\nROC-AUC Score: {roc_log:.4f}")

#makign the confusion matrix for evaluating the performance of our model
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(Y_test , Y_pred_log)
accuracy = accuracy_score(Y_test , Y_pred_log)

#evaluating the performance of the model using stratified k-fold cross validation

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
skf = StratifiedKFold(n_splits = 10 , random_state = 0, shuffle = True)

accuracies_Log = cross_val_score( estimator = Log_classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean_Log = accuracies_Log.mean()
variance_Log = accuracies_Log.std()

print("Accuracy of Logistic Regression:")
print(f"Cross Validation Accuracy: {mean_Log*100:.2f} %")
print(f"Standard Deviation: {variance_Log*100:.2f} %")

# 2. K-NEAREST NEIGHBORS

from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski' , p = 2)
KNN_classifier.fit(X_train, Y_train)

Y_pred_KNN = KNN_classifier.predict(X_test)

#evaluating the performance of the model using confusion matrix and classification report

print("\n================ KNN ================")

cm_knn = confusion_matrix(Y_test , Y_pred_KNN)
print("\nConfusion Matrix:")
print(cm_knn)

print("\nClassification Report:")
print(classification_report(Y_test , Y_pred_KNN))

roc_knn = roc_auc_score(Y_test , Y_pred_KNN)
print(f"\nROC-AUC Score: {roc_knn:.4f}")

# not making confusion matrix and accuracy score for KNN as we will evaluate the performance of the model using stratified k-fold cross validation

accuracies_KNN = cross_val_score(estimator = KNN_classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean_KNN = accuracies_KNN.mean()
variance_KNN = accuracies_KNN.std()

print("Accuracy of K-Nearest Neighbors:")
print(f"Cross Validation Accuracy: {mean_KNN*100:.2f} %")
print(f"Standard Deviation: {variance_KNN*100:.2f} %")

# 3. SUPPORT VECTOR MACHINE

from sklearn.svm import SVC
SVM_classifier = SVC(kernel = 'rbf' , random_state = 0) 
SVM_classifier.fit(X_train , Y_train)

Y_pred_SVM = SVM_classifier.predict(X_test)

#evaluating the performance of the model using confusion matrix and classification report

print("\n================ SVM ================")

cm_svm = confusion_matrix(Y_test , Y_pred_SVM)
print("\nConfusion Matrix:")
print(cm_svm)

print("\nClassification Report:")
print(classification_report(Y_test , Y_pred_SVM))

roc_svm = roc_auc_score(Y_test , Y_pred_SVM)
print(f"\nROC-AUC Score: {roc_svm:.4f}")

accuracies_SVM = cross_val_score(estimator = SVM_classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean_SVM = accuracies_SVM.mean()
variance_SVM = accuracies_SVM.std()

print("Accuracy of Support Vector Machine:")
print(f"Cross Validation Accuracy: {mean_SVM*100:.2f} %")
print(f"Standard Deviation: {variance_SVM*100:.2f} %")

# 4. NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train , Y_train)

Y_pred_NB = NB_classifier.predict(X_test)

#evaluating the performance of the model using confusion matrix and classification report

print("\n================ NAIVE BAYES ================")

cm_nb = confusion_matrix(Y_test , Y_pred_NB)
print("\nConfusion Matrix:")
print(cm_nb)

print("\nClassification Report:")
print(classification_report(Y_test , Y_pred_NB))

roc_nb = roc_auc_score(Y_test , Y_pred_NB)
print(f"\nROC-AUC Score: {roc_nb:.4f}")

accuracies_NB = cross_val_score(estimator = NB_classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean_NB = accuracies_NB.mean()
variance_NB = accuracies_NB.std()
print("Accuracy of Naive Bayes:")
print(f"Cross Validation Accuracy: {mean_NB*100:.2f} %")
print(f"Standard Deviation: {variance_NB*100:.2f} %")

#5. DECISION TREE

from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
DT_classifier.fit(X_train , Y_train)

Y_pred_DT = DT_classifier.predict(X_test)

#evaluating the performance of the model using confusion matrix and classification report

print("\n================ DECISION TREE ================")

cm_dt = confusion_matrix(Y_test , Y_pred_DT)
print("\nConfusion Matrix:")
print(cm_dt)

print("\nClassification Report:")
print(classification_report(Y_test , Y_pred_DT))

roc_dt = roc_auc_score(Y_test , Y_pred_DT)
print(f"\nROC-AUC Score: {roc_dt:.4f}")

accuracies_DT = cross_val_score(estimator = DT_classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean_DT = accuracies_DT.mean()
variance_DT = accuracies_DT.std()

print("Accuracy of Decision Tree:")
print(f"Cross Validation Accuracy: {mean_DT*100:.2f} %")
print(f"Standard Deviation: {variance_DT*100:.2f} %")

# 6. RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10 , criterion = 'entropy' , random_state = 0)
RF_classifier.fit(X_train , Y_train)

Y_pred_RF = RF_classifier.predict(X_test)

#evaluating the performance of the model using confusion matrix and classification report

print("\n================ RANDOM FOREST ================")

cm_rf = confusion_matrix(Y_test , Y_pred_RF)
print("\nConfusion Matrix:")
print(cm_rf)

print("\nClassification Report:")
print(classification_report(Y_test , Y_pred_RF))

roc_rf = roc_auc_score(Y_test , Y_pred_RF)
print(f"\nROC-AUC Score: {roc_rf:.4f}")

accuracies_RF = cross_val_score(estimator = RF_classifier , X = X_train , y = Y_train , cv = skf, scoring= 'accuracy')

mean_RF = accuracies_RF.mean()
variance_RF = accuracies_RF.std()
print("Accuracy of Random Forest:")
print(f"Cross Validation Accuracy: {mean_RF*100:.2f} %")  
print(f"Standard Deviation: {variance_RF*100:.2f} %")



print("\n")
print("============== FINAL COMPARISON ==============")

comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'KNN',
        'SVM',
        'Naive Bayes',
        'Decision Tree',
        'Random Forest'
    ],

    'Cross Val Accuracy (%)': [
        mean_Log*100,
        mean_KNN*100,
        mean_SVM*100,
        mean_NB*100,
        mean_DT*100,
        mean_RF*100
    ],

    'ROC-AUC': [
        roc_log,
        roc_knn,
        roc_svm,
        roc_nb,
        roc_dt,
        roc_rf
    ]
})

print(comparison)