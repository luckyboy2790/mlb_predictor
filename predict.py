# predict.py
import pandas as pd
import joblib

def predict_new_game(input_dict: dict) -> dict:
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    feature_order = joblib.load('model/feature_order.pkl')
    encoders = joblib.load('model/encoders.pkl')  # ✅ load encoders

    df = pd.DataFrame([input_dict])

    # ✅ Apply LabelEncoder to any object (string) fields
    for col in df.select_dtypes(include='object').columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    
    # ✅ Ensure correct order
    missing_cols = [col for col in feature_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing input features: {missing_cols}")
    
    df = df[feature_order]
    X = scaler.transform(df)
    proba = model.predict_proba(X)[0][1]

    return {
        'home_team': input_dict['home_team'],
        'away_team': input_dict['away_team'],
        'win_probability_home_team': round(float(proba), 3),
        'predicted_winner': input_dict['home_team'] if proba >= 0.5 else input_dict['away_team']
    }
