# pipeline/train_model.py

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

def train_model(X_path: str, y_path: str, output_path: str):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()

    print("📊 Training base XGBoost model with calibration...")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    xgb = XGBClassifier(eval_metric='logloss')

    if len(set(y)) < 2 or len(y) < 6:
        print("⚠️ Not enough data for calibration — using raw model.")
        model = xgb
    else:
        model = CalibratedClassifierCV(xgb, method='isotonic', cv=3)

    model.fit(X_scaled, y)

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(model, os.path.join(output_path, 'model.pkl'))
    joblib.dump(scaler, os.path.join(output_path, 'scaler.pkl'))
    joblib.dump(X_scaled.columns.tolist(), os.path.join(output_path, 'feature_order.pkl'))

    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)[:, 1]

    print('✅ Accuracy:', accuracy_score(y, preds))
    print('✅ ROC AUC:', roc_auc_score(y, probas))
    print('✅ Log Loss:', log_loss(y, probas))
