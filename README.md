# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project uses Machine Learning classification algorithms to predict the presence of heart disease based on patient medical attributes. Multiple classification models were trained and evaluated to identify the best-performing algorithm for this healthcare prediction task.

The project demonstrates data preprocessing, feature engineering, model training, evaluation, and comparison of different machine learning classifiers.

---

## 📊 Dataset

**Dataset:** `heart.csv`

### Features include:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol Level
* Maximum Heart Rate
* Exercise-Induced Angina
* Other clinical parameters

### Target Variable

* **0** → No Heart Disease
* **1** → Heart Disease Present

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## 🧠 Machine Learning Algorithms Compared

The following classification algorithms were implemented and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Naive Bayes
5. Decision Tree
6. Random Forest

---

## 🛠️ Workflow

1. Load dataset using Pandas
2. Split features (X) and target (Y)
3. Encode categorical variables using OneHotEncoder
4. Split dataset into Training and Test sets
5. Apply feature scaling using StandardScaler
6. Train multiple classification models
7. Evaluate performance using Cross Validation
8. Compare model accuracy and stability

---

## 📈 Model Performance

| Model                  | Cross Validation Accuracy | Standard Deviation |
| ---------------------- | ------------------------- | ------------------ |
| Logistic Regression    | 87.46%                    | 4.11%              |
| K-Nearest Neighbors    | 86.09%                    | 2.70%              |
| Support Vector Machine | 86.50%                    | 3.67%              |
| Naive Bayes            | 86.37%                    | 2.39%              |
| Decision Tree          | 79.29%                    | 3.92%              |
| Random Forest          | 85.69%                    | 5.05%              |

---

## 🏆 Best Performing Model

**Logistic Regression** achieved the highest cross-validation accuracy of **87.46%**, making it the best-performing model for this dataset.

Additionally, **Naive Bayes** and **KNN** demonstrated lower standard deviation values, indicating more stable performance across folds.

---

## 🔍 Cross Validation

To ensure reliable evaluation, **Stratified K-Fold Cross Validation (k=10)** was used.

Benefits:

* Preserves class distribution in each fold
* Reduces evaluation bias
* Provides a more robust estimate of model performance

---

## 🚀 Future Improvements

* Hyperparameter Tuning using GridSearchCV
* Feature Selection Techniques
* Ensemble Learning Methods
* Model Deployment using Flask or Streamlit
* Interactive Web Application

---

## 👨‍💻 Author

**Harish Jakhar**

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
