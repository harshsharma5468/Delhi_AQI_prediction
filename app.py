from pathlib import Path
import joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from aqi_calculator import calculate_aqi

def forecast_category(aqi):
    return next(label for upper, label in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"), (500, "Severe")] if aqi <= upper)
from data_quality import quality_report
st.set_page_config(page_title="Delhi AQI Intelligence",page_icon="🌫️",layout="wide")
st.title("🌫️ Delhi AQI Intelligence")
st.caption("CPCB-style AQI estimate • multi-point OpenWeather proxy • uncertainty-aware one-hour forecast")
@st.cache_data(ttl=300)
def load_data():
    d=pd.read_csv("aqi_data.csv"); d["datetime_utc"]=pd.to_datetime(d["datetime_utc"],utc=True,errors="coerce")
    return d.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").drop_duplicates("datetime_utc")
def feature_row(d):
    q=d.copy(); q["aqi"]=q.apply(lambda r:calculate_aqi(r)["aqi"],axis=1); row=d.iloc[-1].copy(); h=row.datetime_utc.hour; w=row.datetime_utc.dayofweek
    row["hour_sin"]=np.sin(2*np.pi*h/24); row["hour_cos"]=np.cos(2*np.pi*h/24); row["dow_sin"]=np.sin(2*np.pi*w/7); row["dow_cos"]=np.cos(2*np.pi*w/7)
    row["aqi_lag_1"]=q.iloc[-2].aqi; row["aqi_lag_3"]=q.iloc[-4].aqi; row["aqi_rolling_3"]=q.iloc[-4:-1].aqi.mean()
    return pd.DataFrame([row])
def show_aqi(result,pm25):
    a,b,c,d=st.columns(4); a.metric("CPCB-style AQI",result["aqi"]); b.metric("Health category",result["category"]); c.metric("Dominant pollutant",result["dominant_pollutant"]); d.metric("PM2.5",f"{pm25:.1f} µg/m³"); st.progress(result["aqi"]/500)
try: df=load_data()
except FileNotFoundError: st.error("Live dataset missing."); st.stop()
latest=df.iloc[-1]; current=calculate_aqi(latest); report=quality_report(df)
st.caption("Latest: %s UTC • %s • Data quality: %s (%s/100)" % (latest.datetime_utc.strftime("%d %b %Y %H:%M"),latest.get("location_name","Delhi"),report["status"].title(),report["score"]))
overview,forecast,quality,manual=st.tabs(["Overview","Forecast","Data quality","AQI calculator"])
with overview:
    show_aqi(current,float(latest.pm25)); h=df.tail(240).copy(); h["AQI"]=h.apply(lambda r:calculate_aqi(r)["aqi"],axis=1)
    fig=px.line(h,x="datetime_utc",y="AQI",markers=True,title="Recent CPCB-style AQI estimate"); fig.update_yaxes(range=[0,500]); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame({"Pollutant":list(current["sub_indices"]),"Sub-index":list(current["sub_indices"].values())}).sort_values("Sub-index",ascending=False),hide_index=True,use_container_width=True)
with forecast:
    st.subheader("One-hour AQI forecast"); path=Path("models/aqi_forecaster.pkl")
    if not path.exists() or len(df)<4: st.info("Forecast model is being prepared by the scheduled pipeline.")
    else:
        try:
            art=joblib.load(path); x=feature_row(df)[art["features"]]; point=int(np.clip(round(art["point_model"].predict(x)[0]),0,500)); low=int(np.clip(round(art["lower_model"].predict(x)[0]),0,500)); high=int(np.clip(round(art["upper_model"].predict(x)[0]),0,500)); cat=forecast_category(point)
            a,b,c,d=st.columns(4); a.metric("Forecast AQI",point); b.metric("80% interval","%d–%d"%(min(low,high),max(low,high))); c.metric("Forecast category",cat); d.metric("Temporal MAE",art["metadata"]["metrics"]["mae"]); st.caption("Time-ordered validation only. This is an estimate, not official CPCB AQI.")
        except Exception: st.warning("Forecast model is refreshing for the deployed runtime. Refresh after the workflow completes.")
with quality:
    st.json(report)
    if "stations_used" in latest: st.metric("Aggregation coverage","%s/%s points"%(int(latest.stations_used),int(latest.stations_requested)))
    st.dataframe(df.tail(100).sort_values("datetime_utc",ascending=False),hide_index=True,use_container_width=True)
with manual:
    st.caption("All inputs are µg/m³, including CO, matching OpenWeather API response."); vals={}; fields=["pm25","pm10","no2","so2","o3","co","nh3"]; cols=st.columns(4)
    for i,f in enumerate(fields):
        with cols[i%4]: vals[f]=st.number_input(f.upper().replace("PM25","PM2.5"),min_value=0.0,value=float(latest.get(f,0)),step=1.0)
    if st.button("Calculate AQI",type="primary"): show_aqi(calculate_aqi(vals),vals["pm25"])
