import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cardio_train.csv', sep=';') 

df['age_years'] = (df['age'] / 365.25).round().astype(int)

df_clean = df[
    (df['age_years'] >= 10) & (df['age_years'] <= 100) &
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 160) &
    (df['ap_hi'] >= 60) & (df['ap_hi'] <= 240) &
    (df['ap_hi'] > df['ap_lo']) & 
    (df['height'] >= 100) & (df['height'] <= 250) & 
    (df['weight'] >= 30) & (df['weight'] <= 200)
].copy()

df_clean['bmi'] = df_clean['weight'] / ((df_clean['height'] / 100) ** 2)

df_clean['pulse_pressure'] = df_clean['ap_hi'] - df_clean['ap_lo']

features = ['age_years', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo','pulse_pressure', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
target = 'cardio'

X = df_clean[features]
y = df_clean[target]

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)
y_prob = classifier.predict_proba(X_test_scaled)[:, 1] 

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")






