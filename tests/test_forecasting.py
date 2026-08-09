import numpy as np
import pandas as pd

from train_forecaster import FEATURES, build_feature_frame, hourly_observations


def sample(hours=40):
    timestamps = pd.date_range("2026-01-01", periods=hours * 2, freq="30min", tz="UTC")
    values = np.arange(len(timestamps), dtype=float)
    return pd.DataFrame({"datetime_utc": timestamps, "co": 500 + values,
        "no": 2 + values / 100, "no2": 30 + values / 20, "o3": 40 + values / 30,
        "so2": 5 + values / 100, "pm25": 50 + values / 10,
        "pm10": 90 + values / 8, "nh3": 8 + values / 100})


def test_hourly_observations_use_robust_median():
    hourly = hourly_observations(sample())
    assert len(hourly) == 40
    assert hourly.iloc[0].pm25 == 50.05


def test_feature_frame_has_lags_and_exact_next_hour_target():
    features = build_feature_frame(sample())
    assert not features.empty
    assert set(FEATURES).issubset(features.columns)
    assert (features.datetime_utc.diff().dropna() == pd.Timedelta(hours=1)).all()
    assert features.iloc[0].aqi_lag_24 >= 0


def test_gap_is_not_used_as_next_hour_target():
    raw = sample(50)
    missing = raw.datetime_utc.between(pd.Timestamp("2026-01-02 06:00", tz="UTC"), pd.Timestamp("2026-01-02 06:59", tz="UTC"))
    features = build_feature_frame(raw[~missing])
    assert pd.Timestamp("2026-01-02 05:00", tz="UTC") not in set(features.datetime_utc)
