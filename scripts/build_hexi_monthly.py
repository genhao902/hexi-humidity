import os
import time
import requests
import pandas as pd

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "Wuwei":   {"lat": 37.934246, "lon": 102.638931},
    "Zhangye": {"lat": 38.934170, "lon": 100.451670},
    "Jiuquan": {"lat": 39.733400, "lon": 98.494300},
}

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

HOURLY_VARS = [
    "relative_humidity_2m",
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "pressure_msl"
]


def fetch_hourly(city, lat, lon):
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto"
    }

    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    js = r.json()

    hourly = js["hourly"]
    df = pd.DataFrame()
    df["time"] = pd.to_datetime(hourly["time"])

    for v in HOURLY_VARS:
        df[v] = hourly[v]

    df["city"] = city
    return df


def hourly_to_monthly(df):
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    monthly = df.groupby(["city","year","month"], as_index=False).mean()

    return monthly[[
        "city","year","month",
        "temperature_2m","pressure_msl","wind_speed_10m",
        "dew_point_2m","relative_humidity_2m"
    ]]


all_monthly = []

for city, info in CITIES.items():
    print(f"Fetching: {city}")
    hourly = fetch_hourly(city, info["lat"], info["lon"])
    monthly = hourly_to_monthly(hourly)
    all_monthly.append(monthly)

df = pd.concat(all_monthly, ignore_index=True)

df.to_csv("hexi_monthly_clean.csv", index=False, encoding="utf-8-sig")
print("Saved hexi_monthly_clean.csv")

print(df)
print(df.head())
print(df.tail())