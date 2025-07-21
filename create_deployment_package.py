import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import zipfile
import os

# Step 1: Load data
df = pd.read_csv("bankrupt_clean.csv")

X = df.drop(columns=['class_yn'])
y = df['class_yn']

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 3: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 4: Logistic Regression
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# Step 5: Save model & scaler
joblib.dump(clf, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Step 6: Create 6 prediction CSVs
# Use mean & std to adjust risk levels
X_mean = X.mean()
X_std = X.std()

for level in range(1, 7):
    risk_sample = X_mean + (level - 3) * 0.5 * X_std  # levels 1–6 around mean
    risk_sample = risk_sample.to_frame().T
    risk_sample_scaled = scaler.transform(risk_sample)
    pred = clf.predict(risk_sample_scaled)[0]
    prob = clf.predict_proba(risk_sample_scaled)[0,1]
    
    risk_sample['predicted_class'] = pred
    risk_sample['predicted_prob'] = prob
    
    risk_sample.to_csv(f"prediction_risk_level_{level}.csv", index=False)

# Step 7: Zip everything
with zipfile.ZipFile("deployment_package.zip", 'w') as zipf:
    zipf.write("best_model.pkl")
    zipf.write("scaler.pkl")
    for level in range(1,7):
        fname = f"prediction_risk_level_{level}.csv"
        zipf.write(fname)

print("✅ Deployment package created: deployment_package.zip")
