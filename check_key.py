import requests
import time

API_KEY = "0c45b132f394143a41d114722b459437"
URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=28.61&lon=77.20&appid={API_KEY}"

print("📡 Monitoring OpenWeather activation... (Ctrl+C to stop)")

while True:
    response = requests.get(URL)
    if response.status_code == 200:
        print("✅ SUCCESS! Your key is now ACTIVE.")
        print("Now you can run 'python fetch_api.py' and 'streamlit run app.py'")
        break
    else:
        print(f"⏳ Status {response.status_code}: Key still activating... checking again in 5 mins.")
        time.sleep(300) # Check every 5 minutes