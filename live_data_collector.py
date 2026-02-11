import requests
import pandas as pd
from datetime import datetime

API_KEY = "2c5465d7b56d3e606b26db0d0d257f1bf3e4a160652b5ad1e96445664d299aca"
# 8114 is the ID for Anand Vihar, Delhi
URL = "https://api.openaq.org/v3/locations/8114/latest"
HEADERS = {"X-API-Key": API_KEY, "Accept": "application/json"}

print("Fetching 2026 Delhi Data...")

try:
    r = requests.get(URL, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    
    new_rows = []
    for item in results:
        # OpenAQ v3 uses 'datetime' -> 'utc' for the timestamp
        new_rows.append({
            "date": item.get("datetime", {}).get("utc"),
            "value": item.get("value"),
            "sensorsId": item.get("sensorsId")
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Try to load existing file, but only keep 2026 data
        try:
            old_df = pd.read_csv("aqi_data.csv")
            old_df['date'] = pd.to_datetime(old_df['date'])
            # Only keep rows from 2026
            old_df = old_df[old_df['date'].dt.year == 2026]
            final_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['date'])
        except:
            final_df = new_df

        final_df.to_csv("aqi_data.csv", index=False)
        print(f"✅ Success! CSV updated with data from: {new_rows[0]['date']}")
    else:
        print("❌ API returned no data for today.")

except Exception as e:
    print(f"❌ Error: {e}. The API endpoint might have changed or your key is inactive.")