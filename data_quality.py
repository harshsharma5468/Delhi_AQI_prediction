import pandas as pd
POLLUTANTS = ["co","no","no2","o3","so2","pm25","pm10","nh3"]

def quality_report(df):
    if df.empty: return {"status":"unavailable","score":0,"rows":0}
    d = df.copy()
    d["datetime_utc"] = pd.to_datetime(d["datetime_utc"], utc=True, errors="coerce")
    latest = d["datetime_utc"].max()
    stale = round((pd.Timestamp.now(tz="UTC")-latest).total_seconds()/3600, 1)
    missing = [p for p in POLLUTANTS if p not in d or d[p].tail(24).isna().mean() > .2]
    score = max(0, 100 - len(missing)*10 - (20 if stale > 3 else 0))
    return {"status":"healthy" if score >= 85 else "degraded" if score >= 60 else "unreliable", "score":score, "rows":len(d), "stale_hours":stale, "missing":missing}
