# Loan Pipeline

This project provides a full **Loan Approval Machine Learning pipeline** ready to run in Google Colab.

## Features
- Downloads a public loan dataset automatically (with fallbacks if the primary URL fails).
- Cleans and preprocesses the data (handling missing values, categorical encoding, scaling).
- Builds and evaluates baseline models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Performs quick hyperparameter tuning with GridSearchCV.
- Generates evaluation metrics (accuracy, precision, recall, F1) and confusion matrices.

## How to Run
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `loan_pipeline.py` file.
3. Run:
   ```bash
   !python loan_pipeline.py
