import os
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
import streamlit as st
from aqi_calculator import calculate_aqi

st.set_page_config(page_title="Delhi AQI", page_icon="🌫️", layout="wide")
st.title("🌫️ Delhi AQI — CPCB-style estimate")
st.caption("Current AQI is calculated from pollutant readings using Indian CPCB breakpoints. It is not the OpenWeather 1–5 AQI scale. Official CPCB AQI uses station-specific averaging windows.")

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv("aqi_data.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.sort_values("datetime_utc").drop_duplicates("datetime_utc")

def render_result(reading, title):
    result = calculate_aqi(reading)
    st.subheader(title)
    a, b, c, d = st.columns(4)
    a.metric("AQI", result["aqi"])
    b.metric("Category", result["category"])
    c.metric("Dominant pollutant", result["dominant_pollutant"])
    d.metric("PM2.5", f'{float(reading.get("pm25", 0)):.1f} µg/m³')
    st.progress(result["aqi"] / 500)
    st.caption("Sub-indices: " + " · ".join(f"{p.upper()}: {v}" for p, v in result["sub_indices"].items()))
    return result

try:
    df = load_data()
except FileNotFoundError:
    st.error("aqi_data.csv is missing. Run fetch_api.py first.")
    st.stop()

latest = df.iloc[-1]
st.caption(f"Latest stored observation: {latest.datetime_utc.strftime('%d %b %Y %H:%M UTC')} · Source point: {latest.get('location_name', 'Delhi')}. This is not a full-city station average.")
render_result(latest, "Current AQI estimate from latest reading")

st.divider()
st.subheader("Manual AQI calculation")
st.caption("Enter a current reading to calculate AQI. CO input is in µg/m³, matching OpenWeather data.")
defaults = {key: float(latest.get(key, 0)) for key in ["pm25","pm10","no2","so2","o3","co","nh3"]}
cols = st.columns(4)
manual = {}
for i, (key, value) in enumerate(defaults.items()):
    with cols[i % 4]:
        manual[key] = st.number_input(key.upper().replace("PM25", "PM2.5"), min_value=0.0, value=value, step=1.0)
if st.button("Calculate AQI", type="primary"):
    render_result(manual, "AQI for entered reading")

st.divider()
st.subheader("One-hour AQI forecast")
model_path = Path("models/aqi_forecaster.pkl")
if not model_path.exists():
    st.info("Forecast model has not been trained yet. After collecting sufficient consecutive data, run: python train_forecaster.py")
else:
    artifact = joblib.load(model_path)
    history = df.copy()
    history["aqi"] = history.apply(lambda row: calculate_aqi(row)["aqi"], axis=1)
    if len(history) < 4:
        st.warning("Need at least four readings for lag features.")
    else:
        row = latest.copy()
        row["hour"] = latest.datetime_utc.hour
        row["dayofweek"] = latest.datetime_utc.dayofweek
        row["month"] = latest.datetime_utc.month
        row["aqi_lag_1"] = history.iloc[-2].aqi
        row["aqi_lag_3"] = history.iloc[-4].aqi
        pred = max(0, min(500, round(float(artifact["model"].predict(pd.DataFrame([row])[artifact["features"]])[0]))))
        forecast = calculate_aqi({"pm25": 0})
        category = next((label for upper, label, _ in [(50,"Good","#16a34a"),(100,"Satisfactory","#ca8a04"),(200,"Moderate","#ea580c"),(300,"Poor","#dc2626"),(400,"Very Poor","#7e22ce"),(500,"Severe","#7f1d1d")] if pred <= upper), "Severe")
        st.metric("Forecast AQI (next valid hourly observation)", pred, help=f"Temporal test MAE: {artifact['metrics']['mae']}; RMSE: {artifact['metrics']['rmse']}")
        st.write(f"Forecast category: **{category}**")
        st.caption("A forecast is not shown as official AQI. Its performance must be judged on future, time-ordered data.")
