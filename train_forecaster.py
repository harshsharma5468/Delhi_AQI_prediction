"""Train a genuine one-hour-ahead CPCB AQI forecast from collected data."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from aqi_calculator import calculate_aqi

DATA_PATH = Path("aqi_data.csv")
MODEL_PATH = Path("models/aqi_forecaster.pkl")
FEATURES = ["co","no","no2","o3","so2","pm25","pm10","nh3","hour","dayofweek","month","aqi_lag_1","aqi_lag_3"]

def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc")
    df["aqi"] = df.apply(lambda r: calculate_aqi(r)["aqi"], axis=1)
    df["hour"] = df.datetime_utc.dt.hour
    df["dayofweek"] = df.datetime_utc.dt.dayofweek
    df["month"] = df.datetime_utc.dt.month
    df["aqi_lag_1"] = df.aqi.shift(1)
    df["aqi_lag_3"] = df.aqi.shift(3)
    # A gap over 2 hours means the next row is not a one-hour forecast target.
    next_gap = df.datetime_utc.shift(-1) - df.datetime_utc
    df["target_aqi_1h"] = df.aqi.shift(-1)
    return df[(next_gap <= pd.Timedelta(hours=2))].dropna(subset=FEATURES + ["target_aqi_1h"])

def main() -> None:
    df = prepare_data(DATA_PATH)
    if len(df) < 100:
        raise SystemExit(f"Need at least 100 consecutive hourly readings; found {len(df)} valid training rows.")
    split = list(TimeSeriesSplit(n_splits=4).split(df))[-1]
    train_idx, test_idx = split
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42)
    model.fit(df.iloc[train_idx][FEATURES], df.iloc[train_idx].target_aqi_1h)
    prediction = model.predict(df.iloc[test_idx][FEATURES])
    metrics = {"mae": round(float(mean_absolute_error(df.iloc[test_idx].target_aqi_1h, prediction)), 2), "rmse": round(float(mean_squared_error(df.iloc[test_idx].target_aqi_1h, prediction) ** 0.5), 2), "test_rows": len(test_idx)}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
