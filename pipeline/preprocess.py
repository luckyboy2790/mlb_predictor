import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preprocess_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = df.dropna()
    
    X = df.drop(columns=['result'])
    y = df['result']

    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(encoders, os.path.join(output_path, 'encoders.pkl'))
    joblib.dump(scaler, os.path.join(output_path, 'scaler.pkl'))
    X_scaled.to_csv(os.path.join(output_path, 'X.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y.csv'), index=False)
