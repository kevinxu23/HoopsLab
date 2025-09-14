from __future__ import annotations
import math
import pandas as pd
import numpy as np

EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance via the haversine formula (km)."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R_KM * c

def safe_div(num, den):
    return np.where(den == 0, np.nan, num / den)

def as_date(s: pd.Series|pd.DatetimeIndex) -> pd.Series:
    return pd.to_datetime(s).dt.tz_localize(None)

TEAM_ABBREV_FIX = {
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
    "New York Knicks": "NYK", "New Orleans Pelicans": "NOP",
    "Golden State Warriors": "GSW", "Portland Trail Blazers": "POR",
}

def canon_team(name_or_abbrev: str) -> str:
    x = name_or_abbrev.strip()
    return TEAM_ABBREV_FIX.get(x, x)
