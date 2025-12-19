import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib


# The dataset is comma-separated. Use the correct separator so columns like 'age' are parsed.
df = pd.read_csv('cardio_train.csv', sep=',')

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

# Quy ước: 
#  1: Tăng giá trị -> Tăng nguy cơ bệnh
# -1: Tăng giá trị -> Giảm nguy cơ bệnh
#  0: Không ép buộc (để model tự học)

# Thứ tự features: 
# ['age_years', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo','pulse_pressure', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

constraints = (
    1,  # age_years: Càng già càng dễ bệnh
    0,  # gender: Không ép (tùy dữ liệu)
    0,  # height: Không ép
    1,  # weight: Càng nặng càng dễ bệnh
    0,  # bmi: Dữ liệu hình J/U (Không ép)
    1,  # ap_hi: Huyết áp cao -> bệnh
    1,  # ap_lo: Huyết áp cao -> bệnh
    0,  # pulse_pressure: Dữ liệu hình J/U (Không ép)
    1,  # cholesterol: Cao -> bệnh
    1,  # gluc: Đường huyết cao -> bệnh
    1,  # smoke: Hút thuốc (0->1) -> Tăng nguy cơ (QUAN TRỌNG)
    1,  # alco: Rượu bia (0->1) -> Tăng nguy cơ
    -1  # active: Vận động (0->1) -> Giảm nguy cơ (QUAN TRỌNG)
)

# train model
model = xgb.XGBClassifier(
    n_estimators=200,       # Số lượng cây (càng nhiều càng kỹ nhưng lâu)
    learning_rate=0.05,     # Tốc độ học
    max_depth=6,            # Độ sâu của cây
    monotone_constraints=constraints, # đặt ràng buộc
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("-" * 30)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Lưu model
joblib.dump(model, 'cardio_model.joblib')
print("Xong")

