"""Hourly multi-point Delhi AQI proxy collector."""
import logging, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests

CSV_PATH=Path("aqi_data.csv")
API_KEY=os.environ.get("OPENWEATHER_API_KEY")
URL="https://api.openweathermap.org/data/2.5/air_pollution"
POINTS={"Central":(28.6139,77.2090),"North":(28.7041,77.1025),"South":(28.5355,77.2510),"East":(28.6328,77.2950),"West":(28.6517,77.1119),"North East":(28.7180,77.2700)}
FIELDS=["co","no","no2","o3","so2","pm25","pm10","nh3"]
logging.basicConfig(level=logging.INFO)
log=logging.getLogger(__name__)

def fetch_one(name, coords):
    if not API_KEY: return None
    try:
        payload=requests.get(URL,params={"lat":coords[0],"lon":coords[1],"appid":API_KEY},timeout=20).json()["list"][0]
        c=payload["components"]
        return {"station_proxy":name,"aqi_owm":payload["main"]["aqi"],**{x:c.get("pm2_5" if x=="pm25" else x) for x in FIELDS}}
    except Exception as exc:
        log.warning("%s failed: %s",name,exc); return None

def fetch_current_pollution():
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=[f.result() for f in as_completed([pool.submit(fetch_one,n,c) for n,c in POINTS.items()]) if f.result()]
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows)
    return pd.DataFrame([{"datetime_utc":pd.Timestamp.now(tz="UTC").floor("h"),"location_name":"Delhi multi-point median","source":"OpenWeather","aggregation":"median","stations_requested":len(POINTS),"stations_used":len(d),"data_completeness":round(float(d[FIELDS].notna().mean().mean()*100),1),"aqi_owm":float(d.aqi_owm.median()),**{x:float(d[x].median()) for x in FIELDS}}])

def fetch_and_update():
    old=pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()
    new=fetch_current_pollution()
    if new.empty: return old
    d=pd.concat([old,new],ignore_index=True)
    d["datetime_utc"]=pd.to_datetime(d["datetime_utc"],utc=True)
    d=d.drop_duplicates("datetime_utc",keep="last").sort_values("datetime_utc")
    d.to_csv(CSV_PATH,index=False); return d

if __name__=="__main__": fetch_and_update()
