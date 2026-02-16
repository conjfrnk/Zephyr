"""
Zephyr - Weather API integration
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

from functools import lru_cache
import requests


@lru_cache(maxsize=128)
def wx(lat, lon):
    """Fetch current weather from NWS API for a given lat/lon."""
    meta_url = f"https://api.weather.gov/points/{lat},{lon}"
    try:
        meta_resp = requests.get(
            meta_url, headers={"User-Agent": "Zephyr/0.1"}, timeout=8
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        hourly_url = meta["properties"]["forecastHourly"]
        hourly_resp = requests.get(
            hourly_url, headers={"User-Agent": "Zephyr/0.1"}, timeout=8
        )
        hourly_resp.raise_for_status()

        p = hourly_resp.json()["properties"]["periods"][0]
        return {
            "temp_f": p["temperature"],
            "short": p["shortForecast"],
            "wind_mph": float(p["windSpeed"].split()[0]),
        }
    except requests.exceptions.RequestException as e:
        print(f"Weather API error: {e}")
        return {"temp_f": 0, "short": "N/A", "wind_mph": 0}
