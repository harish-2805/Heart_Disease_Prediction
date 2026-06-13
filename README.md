❤️ Heart Disease Prediction using Machine Learning

This project is a Machine Learning classification model that predicts the presence of heart disease based on patient medical attributes. The model is built using Python and scikit-learn, and uses Logistic Regression for classification.

📌 Project Overview

Heart disease is one of the leading causes of death worldwide. This project aims to build a predictive model that can assist in early detection using machine learning techniques.

The model processes patient data, applies preprocessing techniques, and trains a classification algorithm to predict whether a person is likely to have heart disease.

📊 Dataset
Dataset used: heart.csv
Features include medical attributes such as:
Age
Sex
Chest pain type
Blood pressure
Cholesterol level
And other clinical parameters
Target variable:
0 → No heart disease
1 → Heart disease present
⚙️ Technologies Used
Python 🐍
Pandas
NumPy
Matplotlib
scikit-learn
🧠 Machine Learning Model
Algorithm used: Logistic Regression
Model type: Binary Classification
Library: scikit-learn
Why Logistic Regression?

It is simple, efficient, and works well for binary classification problems like disease prediction.

🛠️ Workflow
Load dataset using Pandas
Split features (X) and target (Y)
Encode categorical features using OneHotEncoder
Split data into training and testing sets (80/20)
Feature scaling using StandardScaler
Train Logistic Regression model
Make predictions
Evaluate model performance
Cross-validation using Stratified K-Fold
📈 Model Evaluation
Accuracy: ~87%
Cross-validation Accuracy: ~87%
Standard Deviation: ~4%

This shows the model is fairly stable and performs consistently across different data splits.

🔍 Cross Validation

To ensure model reliability, Stratified K-Fold Cross Validation (k=10) was used.
This helps in reducing bias and variance in evaluation.

👨‍💻 Author

Harish Jakhar

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
