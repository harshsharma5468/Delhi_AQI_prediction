import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURES = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3', 'hour', 'day', 'month']

def train_model():
    if not os.path.exists('aqi_data.csv'):
        return

    df = pd.read_csv('aqi_data.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df['hour'] = df['datetime_utc'].dt.hour
    df['day'] = df['datetime_utc'].dt.day
    df['month'] = df['datetime_utc'].dt.month

    target = 'pm25'
    df = df.dropna(subset=FEATURES + [target])

    if len(df) < 10:
        print(f"Warning: Only {len(df)} rows found. Augmenting data for training...")
        df = pd.concat([df] * 10, ignore_index=True)

    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Success: Model and scaler trained and saved to models/")

if __name__ == "__main__":
    train_model()