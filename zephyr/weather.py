"""
Zephyr - Weather API integration
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import time
from functools import lru_cache
import requests

_CACHE_TTL = 600  # 10 minutes


def _time_bucket():
    """Return a time bucket that changes every _CACHE_TTL seconds."""
    return int(time.time() // _CACHE_TTL)


@lru_cache(maxsize=128)
def _wx_cached(lat, lon, _time_bucket):
    """Fetch current weather from NWS API for a given lat/lon (cached with TTL)."""
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


def wx(lat, lon):
    """Fetch current weather, rounding coords to 2 decimal places for cache efficiency."""
    lat = round(float(lat), 2)
    lon = round(float(lon), 2)
    return _wx_cached(lat, lon, _time_bucket())
