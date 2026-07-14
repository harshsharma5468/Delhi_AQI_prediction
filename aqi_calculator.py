"""CPCB National AQI calculator.

Inputs use OpenWeather units: pollutants are µg/m³ except CO, which is also
received in µg/m³ and converted to mg/m³ for the Indian AQI breakpoints.
This is an instantaneous estimate; official AQI requires CPCB averaging windows.
"""
from __future__ import annotations
from math import ceil
from typing import Mapping

BREAKPOINTS = {
    "pm25": [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
    "pm10": [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
    "no2": [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,1000,401,500)],
    "so2": [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)],
    "o3": [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)],
    "co": [(0,1.0,0,50),(1.1,2.0,51,100),(2.1,10.0,101,200),(10.1,17.0,201,300),(17.1,34.0,301,400),(34.1,50.0,401,500)],
    "nh3": [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1201,1800,301,400),(1801,2000,401,500)],
}
CATEGORIES = [(50,"Good","#16a34a"),(100,"Satisfactory","#ca8a04"),(200,"Moderate","#ea580c"),(300,"Poor","#dc2626"),(400,"Very Poor","#7e22ce"),(500,"Severe","#7f1d1d")]

def sub_index(value: float, pollutant: str) -> int | None:
    if value is None:
        return None
    value = float(value)
    if pollutant == "co":
        value /= 1000.0
    if value < 0:
        return None
    for low, high, i_low, i_high in BREAKPOINTS[pollutant]:
        if low <= value <= high:
            return ceil(((i_high-i_low)/(high-low))*(value-low)+i_low)
    return 500 if value > BREAKPOINTS[pollutant][-1][1] else None

def calculate_aqi(reading: Mapping[str, float]) -> dict:
    indices = {p: sub_index(reading.get(p), p) for p in BREAKPOINTS}
    available = {p: i for p, i in indices.items() if i is not None}
    if not available:
        raise ValueError("No valid pollutant readings were provided.")
    dominant = max(available, key=available.get)
    aqi = available[dominant]
    category, color = next((name, colour) for upper, name, colour in CATEGORIES if aqi <= upper)
    return {"aqi": aqi, "category": category, "color": color, "dominant_pollutant": dominant.upper().replace("PM25","PM2.5"), "sub_indices": available}
