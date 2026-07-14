"""Time-ordered one-hour AQI forecaster with 80% uncertainty interval."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from aqi_calculator import calculate_aqi

BASE=["co","no","no2","o3","so2","pm25","pm10","nh3"]
FEATURES=BASE+["hour_sin","hour_cos","dow_sin","dow_cos","aqi_lag_1","aqi_lag_3","aqi_rolling_3"]
OUT=Path("models/aqi_forecaster.pkl")

def prepare():
    d=pd.read_csv("aqi_data.csv")
    d["datetime_utc"]=pd.to_datetime(d.datetime_utc,utc=True,errors="coerce")
    d=d.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").drop_duplicates("datetime_utc")
    for c in BASE: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["aqi"]=d.apply(lambda r:calculate_aqi(r)["aqi"],axis=1)
    h=d.datetime_utc.dt.hour; w=d.datetime_utc.dt.dayofweek
    d["hour_sin"]=np.sin(2*np.pi*h/24); d["hour_cos"]=np.cos(2*np.pi*h/24)
    d["dow_sin"]=np.sin(2*np.pi*w/7); d["dow_cos"]=np.cos(2*np.pi*w/7)
    d["aqi_lag_1"]=d.aqi.shift(1); d["aqi_lag_3"]=d.aqi.shift(3); d["aqi_rolling_3"]=d.aqi.shift(1).rolling(3).mean()
    d["target"]=d.aqi.shift(-1); gap=d.datetime_utc.shift(-1)-d.datetime_utc
    return d[gap<=pd.Timedelta(hours=2)].dropna(subset=FEATURES+["target"])

def main():
    d=prepare()
    if len(d)<120: raise SystemExit("Need 120 valid consecutive observations, found %d" % len(d))
    tr,te=list(TimeSeriesSplit(n_splits=4).split(d))[-1]; x,y=d[FEATURES],d.target
    point=HistGradientBoostingRegressor(max_iter=350,learning_rate=.045,l2_regularization=1.5,random_state=42).fit(x.iloc[tr],y.iloc[tr])
    low=GradientBoostingRegressor(loss="quantile",alpha=.1,n_estimators=220,learning_rate=.04,max_depth=2,random_state=42).fit(x.iloc[tr],y.iloc[tr])
    high=GradientBoostingRegressor(loss="quantile",alpha=.9,n_estimators=220,learning_rate=.04,max_depth=2,random_state=42).fit(x.iloc[tr],y.iloc[tr])
    p=point.predict(x.iloc[te]); metrics={"mae":round(float(mean_absolute_error(y.iloc[te],p)),2),"rmse":round(float(mean_squared_error(y.iloc[te],p)**.5),2),"coverage_80":round(float(np.mean((y.iloc[te]>=low.predict(x.iloc[te]))&(y.iloc[te]<=high.predict(x.iloc[te])))*100),1),"test_rows":len(te)}
    meta={"schema_version":"2.0","target":"AQI at next valid hourly observation","trained_at_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),"sklearn":sklearn.__version__,"metrics":metrics}
    OUT.parent.mkdir(exist_ok=True); joblib.dump({"point_model":point,"lower_model":low,"upper_model":high,"features":FEATURES,"metadata":meta},OUT); OUT.with_suffix(".json").write_text(json.dumps(meta,indent=2))
if __name__=="__main__": main()
