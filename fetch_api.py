import os
import sys
import logging
import requests
import pandas as pd
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
# Central Coordinates for Delhi (City Average)
LAT, LON = 28.6139, 77.2090  
LOCATION_NAME = "Full Delhi (City Average)"
CSV_PATH = Path("aqi_data.csv")
MIN_YEAR = 2026

# Replace 'your_key_here' with your actual key if not using environment variables
API_KEY = os.environ.get("OPENWEATHER_API_KEY" )
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

def fetch_current_pollution():
    """Fetches real-time pollution data for Full Delhi."""
    url = f"{BASE_URL}?lat={LAT}&lon={LON}&appid={API_KEY}"
    log.info(f"Fetching data for {LOCATION_NAME}...")
    
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 401:
            log.error("API Key Unauthorized. Your key may still be activating (takes 30-120 mins) or is incorrect.")
            return pd.DataFrame()
        
        response.raise_for_status()
        data = response.json()

        if "list" not in data or not data["list"]:
            return pd.DataFrame()

        item = data["list"][0]
        comp = item["components"]
        
        row = {
            "datetime_utc": pd.to_datetime(item["dt"], unit='s', utc=True),
            "location_name": LOCATION_NAME,
            "aqi_owm": item["main"]["aqi"], 
            "co": comp.get("co"),
            "no": comp.get("no"),
            "no2": comp.get("no2"),
            "o3": comp.get("o3"),
            "so2": comp.get("so2"),
            "pm25": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "nh3": comp.get("nh3")
        }
        return pd.DataFrame([row])
    except Exception as e:
        log.error(f"Network error: {e}")
        return pd.DataFrame()

def fetch_and_update():
    new_df = fetch_current_pollution()
    if new_df.empty: return load_existing_csv()

    existing_df = load_existing_csv()
    combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True)
    combined.drop_duplicates(subset=["datetime_utc"], keep="last", inplace=True)
    combined = combined[combined["datetime_utc"].dt.year >= MIN_YEAR]
    combined.sort_values("datetime_utc", inplace=True)
    
    combined.to_csv(CSV_PATH, index=False)
    log.info(f"Success! Delhi data updated. Total history: {len(combined)} rows.")
    return combined

def load_existing_csv():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        return df
    return pd.DataFrame()

if __name__ == "__main__":
    fetch_and_update()



