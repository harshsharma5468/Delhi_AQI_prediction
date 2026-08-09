"""Leakage-safe, time-ordered one-hour Delhi AQI forecaster."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from aqi_calculator import calculate_aqi

BASE = ["co", "no", "no2", "o3", "so2", "pm25", "pm10", "nh3"]
LAGS = (1, 2, 3, 6, 12, 24)
FEATURES = BASE + [
    "aqi_current", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    *[f"aqi_lag_{lag}" for lag in LAGS],
    "aqi_rolling_3", "aqi_rolling_6", "aqi_rolling_24",
    "aqi_trend_1", "aqi_trend_3", "pm25_trend_1", "pm10_trend_1",
]
OUT = Path("models/aqi_forecaster.pkl")


def hourly_observations(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one robust observation per UTC hour without filling missing hours."""
    data = raw.copy()
    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"], utc=True, errors="coerce")
    data = data.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    for column in BASE:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return (data.set_index("datetime_utc")[BASE].resample("1h").median()
            .dropna(how="all").reset_index())


def build_feature_frame(raw: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    data = hourly_observations(raw)
    data["aqi_current"] = data.apply(lambda row: calculate_aqi(row)["aqi"], axis=1)
    hour, weekday = data.datetime_utc.dt.hour, data.datetime_utc.dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["dow_cos"] = np.cos(2 * np.pi * weekday / 7)
    for lag in LAGS:
        data[f"aqi_lag_{lag}"] = data.aqi_current.shift(lag)
    for window in (3, 6, 24):
        data[f"aqi_rolling_{window}"] = data.aqi_current.shift(1).rolling(window).mean()
    data["aqi_trend_1"] = data.aqi_current - data.aqi_lag_1
    data["aqi_trend_3"] = data.aqi_current - data.aqi_lag_3
    data["pm25_trend_1"] = data.pm25 - data.pm25.shift(1)
    data["pm10_trend_1"] = data.pm10 - data.pm10.shift(1)
    required = list(FEATURES)
    if include_target:
        data["target"] = data.aqi_current.shift(-1)
        next_gap = data.datetime_utc.shift(-1) - data.datetime_utc
        data.loc[next_gap != pd.Timedelta(hours=1), "target"] = np.nan
        required.append("target")
    return data.dropna(subset=required).reset_index(drop=True)


def point_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04,
        l2_regularization=1.5, early_stopping=False, random_state=42, **params)


def select_point_model(x: pd.DataFrame, y: pd.Series, current: pd.Series) -> tuple[dict, float, list[dict]]:
    candidates = [
        {"max_leaf_nodes": 15, "min_samples_leaf": 20},
        {"max_leaf_nodes": 31, "min_samples_leaf": 20},
        {"max_leaf_nodes": 31, "min_samples_leaf": 40},
        {"max_leaf_nodes": 63, "min_samples_leaf": 30},
    ]
    splitter, scores = TimeSeriesSplit(n_splits=4), []
    for params in candidates:
        predictions = {weight: [] for weight in (0.0, 0.25, 0.5, 0.75, 1.0)}
        actuals = []
        for train_idx, valid_idx in splitter.split(x):
            delta = y.iloc[train_idx] - current.iloc[train_idx]
            model = point_model(params).fit(x.iloc[train_idx], delta)
            predicted_delta = model.predict(x.iloc[valid_idx])
            actuals.extend(y.iloc[valid_idx].to_numpy())
            for weight in predictions:
                predictions[weight].extend(current.iloc[valid_idx].to_numpy() + weight * predicted_delta)
        weight_scores = {weight: float(mean_absolute_error(actuals, values))
                         for weight, values in predictions.items()}
        best_weight = min(weight_scores, key=weight_scores.get)
        scores.append({"params": params, "delta_weight": best_weight,
            "walk_forward_mae": round(weight_scores[best_weight], 3)})
    best = min(scores, key=lambda result: result["walk_forward_mae"])
    return best["params"], best["delta_weight"], scores


def quantile_model(alpha: float) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=350,
        learning_rate=0.035, max_depth=2, min_samples_leaf=15, random_state=42)


def main() -> None:
    data = build_feature_frame(pd.read_csv("aqi_data.csv"))
    if len(data) < 240:
        raise SystemExit(f"Need 240 valid hourly observations, found {len(data)}")
    test_size = max(168, int(len(data) * 0.20))
    train, test = data.iloc[:-test_size], data.iloc[-test_size:]
    x_train, y_train = train[FEATURES], train.target
    x_test, y_test = test[FEATURES], test.target
    selected, delta_weight, cv_scores = select_point_model(x_train, y_train, train.aqi_current)

    calibration_size = max(96, int(len(train) * 0.15))
    proper, calibration = train.iloc[:-calibration_size], train.iloc[-calibration_size:]
    lower_cal = quantile_model(0.10).fit(proper[FEATURES], proper.target)
    upper_cal = quantile_model(0.90).fit(proper[FEATURES], proper.target)
    cal_low, cal_high = lower_cal.predict(calibration[FEATURES]), upper_cal.predict(calibration[FEATURES])
    conformity = np.maximum.reduce([cal_low - calibration.target.to_numpy(),
        calibration.target.to_numpy() - cal_high, np.zeros(len(calibration))])
    conformal_q = float(np.quantile(conformity, 0.90, method="higher"))

    point = point_model(selected).fit(x_train, y_train - train.aqi_current)
    lower, upper = quantile_model(0.10).fit(x_train, y_train), quantile_model(0.90).fit(x_train, y_train)
    prediction = np.clip(test.aqi_current.to_numpy() + delta_weight * point.predict(x_test), 0, 500)
    low = np.clip(lower.predict(x_test) - conformal_q, 0, 500)
    high = np.clip(upper.predict(x_test) + conformal_q, 0, 500)
    low, high = np.minimum(low, high), np.maximum(low, high)
    actual, persistence = y_test.to_numpy(), test.aqi_current.to_numpy()
    model_mae, baseline_mae = mean_absolute_error(actual, prediction), mean_absolute_error(actual, persistence)
    metrics = {
        "mae": round(float(model_mae), 2),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 2),
        "persistence_mae": round(float(baseline_mae), 2),
        "mae_improvement_vs_persistence_pct": round(float((1 - model_mae / max(baseline_mae, 1e-9)) * 100), 1),
        "coverage_80": round(float(np.mean((actual >= low) & (actual <= high)) * 100), 1),
        "test_rows": len(test),
    }
    validation = pd.DataFrame({
        "datetime_utc": test.datetime_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actual": np.round(actual, 2), "predicted": np.round(prediction, 2),
        "lower": np.round(low, 2), "upper": np.round(high, 2),
        "absolute_error": np.round(np.abs(actual - prediction), 2),
    }).tail(168).to_dict("records")
    metadata = {
        "schema_version": "3.0", "target": "AQI at the next clock hour",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "training_rows": len(train), "selected_params": selected,
        "point_prediction": "current AQI + delta_weight * predicted one-hour change",
        "delta_weight": delta_weight,
        "walk_forward_candidates": cv_scores, "conformal_adjustment": round(conformal_q, 3),
        "metrics": metrics,
    }
    artifact = {"point_model": point, "lower_model": lower, "upper_model": upper,
        "features": FEATURES, "metadata": metadata, "conformal_adjustment": conformal_q,
        "delta_weight": delta_weight,
        "validation": validation}
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, OUT)
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
