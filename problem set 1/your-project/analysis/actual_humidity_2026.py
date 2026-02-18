# -*- coding: utf-8 -*-
"""
Fetch Jan–Feb 2026 monthly mean relative humidity for
Wuwei, Zhangye, Jiuquan from Open-Meteo Archive API.

- Uses 'params' dict to avoid malformed URLs (& missing).
- Truncates end_date to today's date (prevents requesting future data).
- Aggregates hourly relative_humidity_2m to monthly means.

Output:
  data/actual_humidity_2026_janfeb.csv
"""
import sys
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import requests

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# City coordinates (consistent with your project)
CITIES = {
    "Wuwei":   {"lat": 37.934246, "lon": 102.638931},
    "Zhangye": {"lat": 38.934170, "lon": 100.451670},
    "Jiuquan": {"lat": 39.733400, "lon":  98.494300},
}

START_DATE_STR = "2026-01-01"  # fixed
REQUEST_END_STR = "2026-02-28" # target upper bound (will be clipped to today)
HOURLY_VAR = "relative_humidity_2m"  # hourly variable name

OUT_PATH = Path("data/actual_humidity_2026_janfeb.csv")


def safe_end_date(target_end_str: str) -> str:
    """Clip end_date to today's date (UTC) to avoid requesting the future."""
    today = date.today().isoformat()
    return min(target_end_str, today)


def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VAR,     # let requests handle encoding
        "timezone": "auto",
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # 打印服务端返回内容，便于定位 400 的具体原因
        print("❌ HTTP error:", e)
        print("Request URL:", r.url)
        print("Server says:", r.text[:500])
        raise

    js = r.json()
    if "hourly" not in js or HOURLY_VAR not in js["hourly"]:
        raise RuntimeError("No hourly humidity data returned for the requested period.")
    df = pd.DataFrame({
        "time": pd.to_datetime(js["hourly"]["time"]),
        "humidity": js["hourly"][HOURLY_VAR],
    })
    return df


def hourly_to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    out = (df.groupby(["year", "month"], as_index=False)["humidity"]
             .mean()
             .rename(columns={"humidity": "humidity_mean"}))
    return out


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    end_date_str = safe_end_date(REQUEST_END_STR)
    rows = []
    for city, meta in CITIES.items():
        print(f"Fetching: {city} ({meta['lat']}, {meta['lon']})  {START_DATE_STR} → {end_date_str}")
        hourly = fetch_hourly(meta["lat"], meta["lon"], START_DATE_STR, end_date_str)
        monthly = hourly_to_monthly_mean(hourly)
        monthly.insert(0, "city", city)
        rows.append(monthly)

    all_monthly = pd.concat(rows, ignore_index=True)
    all_monthly = all_monthly[(all_monthly["year"] == 2026) & (all_monthly["month"].isin([1, 2]))]
    all_monthly.to_csv(OUT_PATH, index=False)

    print("\n✅ Saved:", OUT_PATH.resolve())
    print(all_monthly)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)