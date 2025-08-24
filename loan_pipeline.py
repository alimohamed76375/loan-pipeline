!pip install -q scikit-learn pandas matplotlib

import os
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = Path("/content")
DATA_PATH = DATA_DIR / "loan_train.csv"
urls_to_try = [
    "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv",
    "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Loan-Approval-Prediction.csv",
    "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
]

for url in urls_to_try:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read()
            text = data[:2000].decode(errors='ignore')
            if ',' in text and len(text) > 10:
                with open(DATA_PATH, "wb") as f:
                    f.write(data)
                break
    except Exception:
        pass
else:
    raise RuntimeError("Failed to download dataset")

df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)
display(df.head(8))

def preprocess(df):
    df = df.copy()
    for idc in ["Loan_ID", "LoanId", "loan_id", "LoanId"]:
        if idc in df.columns:
            df.drop(columns=[idc], inplace=True, errors="ignore")
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")
    if {"ApplicantIncome","CoapplicantIncome"}.issubset(df.columns):
        df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
        df["CoapplicantIncome"] = pd.to_numeric(df["CoapplicantIncome"], errors="coerce")
        df["TotalIncome"] = df["ApplicantIncome"].fillna(0) + df["CoapplicantIncome"].fillna(0)
    target_candidates = ["Loan_Status","loan_status","LoanStatus","LoanStatus","Status","Approved","Target","target"]
    target = None
    for t in target_candidates:
        if t in df.columns:
            target = t
            break
    if target is None:
        raise ValueError("Target column not found")
    y = df[target].copy()
    if y.dtype == object:
        y = y.map({"Y":1,"N":0,"Yes":1,"No":0,"Approved":1,"Rejected":0,"1":1,"0":0}).astype(int)
    X = df.drop(columns=[target])
    return X, y

X, y = preprocess(df)
print("Features:", list(X.columns))
print("Target distribution:\n", y.value_counts())

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
if len(num_cols) == 0:
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
            num_cols.append(c)
        except:
            pass
    cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

def fit_and_report(pipeline, name="model"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.show()
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}

pipe_lr = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
res_lr = fit_and_report(pipe_lr, "LogisticRegression (baseline)")

pipe_dt = Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(random_state=42))])
res_dt = fit_and_report(pipe_dt, "DecisionTree (baseline)")

pipe_rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
res_rf = fit_and_report(pipe_rf, "RandomForest (baseline)")

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

param_lr = {"clf__penalty":["l2"], "clf__C":[0.1,1.0,10.0], "clf__solver":["lbfgs"]}
gs_lr = GridSearchCV(Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=2000, random_state=42))]),
                     param_lr, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
gs_lr.fit(X_train, y_train)
print("Best LR params:", gs_lr.best_params_)
best_lr = gs_lr.best_estimator_
res_best_lr = fit_and_report(best_lr, "LogisticRegression (Grid)")

param_rf = {"clf__n_estimators":[100,200], "clf__max_depth":[None,5,10]}
gs_rf = GridSearchCV(Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42))]),
                     param_rf, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)
best_rf = gs_rf.best_estimator_
res_best_rf = fit_and_report(best_rf, "RandomForest (Grid)")

summary = pd.DataFrame([
    {"model":"Logistic-baseline", **res_lr},
    {"model":"DecisionTree-baseline", **res_dt},
    {"model":"RandomForest-baseline", **res_rf},
    {"model":"Logistic-grid", **res_best_lr},
    {"model":"RandomForest-grid", **res_best_rf}
])
print("\nExperiment summary:")
display(summary.sort_values("f1", ascending=False).reset_index(drop=True))
