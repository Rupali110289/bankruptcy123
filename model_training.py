import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, roc_curve, log_loss, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.impute import SimpleImputer

df = pd.read_csv("bankrupt_clean.csv")
print("Original Dataset Size:", df.shape)


np.random.seed(42)
mask = np.random.rand(*df.shape) < 0.05
df = df.mask(mask)

imputer = SimpleImputer(strategy="mean")
df[df.columns] = imputer.fit_transform(df)
df["class_yn"] = df["class_yn"].round().astype(int)

# Add more noise to input features
df[df.columns[:-1]] += np.random.normal(0, 0.4, df[df.columns[:-1]].shape)

# Step 3: Feature and Target Definition
X = df.drop("class_yn", axis=1)
y = df["class_yn"]

# Flip 15% labels to challenge models
flip_idx = np.random.choice(y.index, size=int(0.15 * len(y)), replace=False)
y.loc[flip_idx] = 1 - y.loc[flip_idx]

# Step 4: Train-Test Split (only 30% training data)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.7, random_state=42)

# Step 5: Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model Tuning with GridSearchCV
print("\n\U0001F50D Starting GridSearchCV for All Models...\n")
best_models = {}

# Logistic Regression
grid_lr = GridSearchCV(LogisticRegression(random_state=42), {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}, cv=5, scoring="f1")
grid_lr.fit(X_train_scaled, y_train)
best_models["Logistic Regression"] = grid_lr.best_estimator_
print("Logistic Regression Best Params:", grid_lr.best_params_)

# Random Forest
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}, cv=5, scoring="f1")
grid_rf.fit(X_train, y_train)
best_models["Random Forest"] = grid_rf.best_estimator_
print("Random Forest Best Params:", grid_rf.best_params_)

# SVM
grid_svm = GridSearchCV(SVC(probability=True, random_state=42), {
    "C": [0.1, 1, 10],
    "kernel": ["linear"]
}, cv=5, scoring="f1")
grid_svm.fit(X_train_scaled, y_train)
best_models["SVM"] = grid_svm.best_estimator_
print("SVM Best Params:", grid_svm.best_params_)

# XGBoost
grid_xgb = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=42), {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}, cv=5, scoring="f1")
grid_xgb.fit(X_train, y_train)
best_models["XGBoost"] = grid_xgb.best_estimator_
print("XGBoost Best Params:", grid_xgb.best_params_)

models = best_models

# Step 7: Evaluate Models
results = {}
for name, model in models.items():
    if name in ["SVM", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

# Step 8: Confusion Matrices
for name, res in results.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Step 9: ROC Curves
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# Step 10: Feature Importance
importances_df = []
for model_name in ["XGBoost", "Random Forest"]:
    model = best_models[model_name]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
            "Model": model_name
        })
        importances_df.append(df_imp)
importances_df = pd.concat(importances_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df, x="Importance", y="Feature", hue="Model")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Step 11: SHAP Summary for XGBoost
explainer = shap.Explainer(grid_xgb.best_estimator_, X_test)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, max_display=10)
shap.plots.beeswarm(shap_values, max_display=10)

# Step 12: Classification Report
for name, res in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {res['accuracy']:.2f}")
    print(f"ROC AUC: {res['roc_auc']:.2f}")
    print(classification_report(y_test, res["y_pred"]))

# Step 13: Cross-Validation
print("\nCross-Validation Scores (5-Fold)")
for name, model in models.items():
    print(f"\n{name}")
    if name in ["SVM", "Logistic Regression"]:
        f1_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        acc_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:
        f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        acc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Mean F1: {f1_scores.mean():.4f}, Std: {f1_scores.std():.4f}")
    print(f"Mean Accuracy: {acc_scores.mean():.4f}, Std: {acc_scores.std():.4f}")

# Step 14: Log Loss
print("\nLog Loss for Each Model:")
for name, res in results.items():
    print(f"{name}: Log Loss = {log_loss(y_test, res['y_proba']):.4f}")

# Step 15: Precision-Recall Curve
plt.figure(figsize=(8, 6))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
    ap_score = average_precision_score(y_test, res["y_proba"])
    plt.plot(recall, precision, label=f"{name} (AP = {ap_score:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid()
plt.show()

# Step 16: Final Comparison and Recommendation
model_scores = []
print("\nF1 Scores on Test Set:")
for name, res in results.items():
    f1 = f1_score(y_test, res["y_pred"])
    print(f"{name}: F1 Score = {f1:.4f}")
    model_scores.append((name, res["accuracy"], res["roc_auc"], f1))

score_df = pd.DataFrame(model_scores, columns=["Model", "Accuracy", "ROC AUC", "F1 Score"])
score_df = score_df.sort_values(by="F1 Score", ascending=False)
print("\n\U0001F4C8 Model Comparison Summary:")
print(score_df)

# Step 17: Recommendation
best_model_name = score_df.iloc[0]["Model"]
print(f"\n\u2705 Suggested model for deployment: **{best_model_name}**\n(Reason: Best F1 and ROC AUC performance with generalization)")

# 🚀 Save the trained Logistic Regression model
import joblib

joblib.dump(best_models["Logistic Regression"], 'logistic_model.pkl')
print("✅ Logistic Regression model saved as logistic_model.pkl")
