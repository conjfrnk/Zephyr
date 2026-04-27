"""
Zephyr - Weather API integration
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import math
import re
import time
from functools import lru_cache
import requests

_CACHE_TTL = 600  # 10 minutes


def _failure_sentinel():
    """Return the standard failure-shape dict so callers can branch on `available`."""
    return {
        "temp_f": None,
        "short": "N/A",
        "wind_mph": None,
        "available": False,
    }


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

        wind_str = p.get("windSpeed", "") or ""
        m = re.search(r"(\d+(?:\.\d+)?)", wind_str)
        wind_mph = float(m.group(1)) if m else 0.0

        return {
            "temp_f": float(p["temperature"]),
            "short": p["shortForecast"],
            "wind_mph": wind_mph,
            "available": True,
        }
    except (
        requests.exceptions.RequestException,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
    ) as e:
        print(f"Weather API error: {e}")
        return _failure_sentinel()


def wx(lat, lon):
    """Fetch current weather, rounding coords to 2 decimal places for cache efficiency."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return _failure_sentinel()

    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return _failure_sentinel()

    lat_r = round(lat_f, 2)
    lon_r = round(lon_f, 2)
    return _wx_cached(lat_r, lon_r, _time_bucket())
