import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    if not os.path.exists('aqi_data.csv'):
        return

    df = pd.read_csv('aqi_data.csv')
    
    # Feature Engineering
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df['hour'] = df['datetime_utc'].dt.hour
    df['day'] = df['datetime_utc'].dt.day
    df['month'] = df['datetime_utc'].dt.month

    features = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3', 'hour', 'day', 'month']
    target = 'pm25'
    df = df.dropna(subset=features + [target])

    # --- HANDLING SMALL DATASETS ---
    if len(df) < 10:
        print(f"Warning: Only {len(df)} rows found. Augmenting data for training...")
        # Add 10 dummy rows based on your current averages
        df = pd.concat([df] * 10, ignore_index=True)
    # -------------------------------------------

    X = df[features]
    y = df[target]
    
    # Split with a smaller test size since data is scarce
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42) # Fewer trees for tiny data
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')
    print("Success: Dummy model trained and saved to models/trained_model.pkl")

if __name__ == "__main__":
    train_model()