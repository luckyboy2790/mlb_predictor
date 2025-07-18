import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model(X_path: str, y_path: str, output_path: str):
    # Load the dataset
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📊 Training base XGBoost model with calibration...")

    # Scale the training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # Initialize XGBoost model
    xgb = XGBClassifier(eval_metric='logloss')

    # If we have enough data, calibrate the model
    if len(set(y_train)) < 2 or len(y_train) < 6:
        print("⚠️ Not enough data for calibration — using raw model.")
        model = xgb
    else:
        model = CalibratedClassifierCV(xgb, method='isotonic', cv=3)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Save the model and preprocessing artifacts
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(model, os.path.join(output_path, 'model.pkl'))
    joblib.dump(scaler, os.path.join(output_path, 'scaler.pkl'))
    joblib.dump(X_train_scaled.columns.tolist(), os.path.join(output_path, 'feature_order.pkl'))

    # Test the model on the test set
    test_model(model, scaler, X_test, y_test)

    return model, scaler

def test_model(model, scaler, X_test, y_test):
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Predict the labels for the test set
    y_pred = model.predict(X_test_scaled)
    probas = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the model using accuracy, ROC AUC, and log loss
    accuracy = accuracy_score(y_test, y_pred)
    print('✅ Test Accuracy:', accuracy)

    roc_auc = roc_auc_score(y_test, probas)
    print('✅ Test ROC AUC:', roc_auc)

    log_loss_score = log_loss(y_test, probas)
    print('✅ Test Log Loss:', log_loss_score)

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'log_loss': log_loss_score
    }
