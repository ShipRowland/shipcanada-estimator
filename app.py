import os
import json
import time
import math
import threading
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.stats import iqr

from config import (
    HISTORICAL_URL, COLUMN_MAP, DEFAULT_ORIGIN_COUNTRY, DEFAULT_DEST_COUNTRY,
    MATCH_RADII, BANDS, GEOCODER_USER_AGENT, GEOCODER_TIMEOUT, GEOCODER_SLEEP_S,
    CACHE_DIR, LANES_PARQUET, GEOCODE_JSON, CURRENCY
)

app = Flask(__name__)

# ---------------------------
# Utilities
# ---------------------------

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def haversine_miles(lat1, lon1, lat2, lon2):
    # Haversine formula
    R = 3958.7613  # Earth radius miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def cuft_from_inches(L, W, H):
    return (L * W * H) / 1728.0


def dim_weight_lb(cuft):
    return cuft * 10.0

# ---------------------------
# Data ingest & geocoding
# ---------------------------

_lock = threading.Lock()
_geocode_cache = {}


def load_geocode_cache():
    global _geocode_cache
    ensure_cache_dir()
    if os.path.exists(GEOCODE_JSON):
        with open(GEOCODE_JSON, "r") as f:
            _geocode_cache = json.load(f)
    else:
        _geocode_cache = {}


def save_geocode_cache():
    ensure_cache_dir()
    with open(GEOCODE_JSON, "w") as f:
        json.dump(_geocode_cache, f, indent=2)


def geocode_key(city, region, country):
    return "|".join([str(city or "").strip().lower(),
                     str(region or "").strip().lower(),
                     str(country or "").strip().lower()])


def make_geocoder():
    geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=GEOCODER_TIMEOUT)
    # add a rate limiter to be polite
    return RateLimiter(geolocator.geocode, min_delay_seconds=GEOCODER_SLEEP_S)


def geocode_city(city: str, region: Optional[str], country: Optional[str]) -> Optional[Tuple[float, float]]:
    """Geocode with cache & graceful fallback."""
    if not city:
        return None
    key = geocode_key(city, region, country or "")
    with _lock:
        if key in _geocode_cache:
            return tuple(_geocode_cache[key])

    geo = make_geocoder()
    query = ", ".join([x for x in [city, region, country] if x])
    try:
        loc = geo(query)
    except Exception:
        loc = None

    if loc:
        latlon = (loc.latitude, loc.longitude)
        with _lock:
            _geocode_cache[key] = latlon
            save_geocode_cache()
        return latlon
    return None


def normalize_str(x):
    return None if pd.isna(x) or str(x).strip() == "" else str(x).strip()


def fetch_and_build_dataset(force=False) -> pd.DataFrame:
    """
    Scrape the historical table from HISTORICAL_URL into a normalized lanes DataFrame,
    geocode unique cities, and persist as Parquet.
    """
    ensure_cache_dir()

    if (not force) and os.path.exists(LANES_PARQUET):
        return pd.read_parquet(LANES_PARQUET)

    # read all tables on the page, take the largest one by rows*cols as the historical
    resp = requests.get(HISTORICAL_URL, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(resp.text, flavor="bs4")
    if not tables:
        raise RuntimeError("No HTML tables found on historical URL.")
    df = max(tables, key=lambda t: (t.shape[0] * t.shape[1])).copy()

    # Harmonize columns
    def col(name_key):
        src = COLUMN_MAP.get(name_key)
        return src if (src in df.columns) else None

    df_out = pd.DataFrame()
    df_out["origin_city"] = df[col("origin_city")] if col("origin_city") else np.nan
    df_out["origin_region"] = df[col("origin_region")] if col("origin_region") else np.nan
    df_out["origin_country"] = df[col("origin_country")] if col("origin_country") else DEFAULT_ORIGIN_COUNTRY

    df_out["dest_city"] = df[col("dest_city")] if col("dest_city") else np.nan
    df_out["dest_region"] = df[col("dest_region")] if col("dest_region") else np.nan
    df_out["dest_country"] = df[col("dest_country")] if col("dest_country") else DEFAULT_DEST_COUNTRY

    df_out["equipment"] = df[col("equipment")] if col("equipment") else "Van"
    df_out["subtotal_cad"] = df[col("subtotal_cad")].apply(to_float) if col("subtotal_cad") else np.nan

    if COLUMN_MAP.get("weight_proxy_lb") and COLUMN_MAP["weight_proxy_lb"] in df.columns:
        df_out["weight_proxy_lb"] = df[COLUMN_MAP["weight_proxy_lb"]].apply(to_float)
    else:
        df_out["weight_proxy_lb"] = np.nan  # may be filled later or ignored

    # Normalize strings
    for c in ["origin_city","origin_region","origin_country","dest_city","dest_region","dest_country","equipment"]:
        df_out[c] = df_out[c].apply(normalize_str)

    # Drop rows missing critical fields
    df_out = df_out.dropna(subset=["origin_city", "dest_city", "subtotal_cad"])

    # Geocode unique endpoints
    load_geocode_cache()

    def geocode_row(city, region, country):
        latlon = geocode_city(city, region, country)
        return pd.Series(latlon if latlon else (np.nan, np.nan))

    # origin
    o_geo = df_out[["origin_city","origin_region","origin_country"]].drop_duplicates().copy()
    o_geo[["o_lat","o_lng"]] = o_geo.apply(lambda r: geocode_row(r.origin_city, r.origin_region, r.origin_country), axis=1)
    df_out = df_out.merge(o_geo, on=["origin_city","origin_region","origin_country"], how="left")

    # destination
    d_geo = df_out[["dest_city","dest_region","dest_country"]].drop_duplicates().copy()
    d_geo[["d_lat","d_lng"]] = d_geo.apply(lambda r: geocode_row(r.dest_city, r.dest_region, r.dest_country), axis=1)
    df_out = df_out.merge(d_geo, on=["dest_city","dest_region","dest_country"], how="left")

    # Keep only rows with both geocodes & positive cost
    df_out = df_out[
        (~df_out["o_lat"].isna()) &
        (~df_out["d_lat"].isna()) &
        (df_out["subtotal_cad"] > 0)
    ].reset_index(drop=True)

    df_out.to_parquet(LANES_PARQUET, index=False)
    return df_out

# ---------------------------
# Core Estimator
# ---------------------------


def compute_estimate(payload: dict) -> dict:
    """
    payload:
      {
        "origin": {"city":"Toronto","region":"ON","country":"CA"},
        "destination":{"city":"Vancouver","region":"BC","country":"CA"},
        "dimensions_in":{"length":48,"width":48,"height":48},
        "weight_lb": 600,
        "pieces": 1,
        "equipment": "Van"
      }
    """
    # Parse inputs
    o = payload.get("origin", {})
    d = payload.get("destination", {})
    dims = payload.get("dimensions_in", {})

    L = float(dims.get("length", 0))
    W = float(dims.get("width", 0))
    H = float(dims.get("height", 0))
    weight_lb = float(payload.get("weight_lb", 0))
    pieces = int(payload.get("pieces", 1) or 1)
    equipment = payload.get("equipment", None)

    # Basic input validation
    missing = []
    if not o.get("city"): missing.append("origin.city")
    if not d.get("city"): missing.append("destination.city")
    if L <= 0: missing.append("dimensions_in.length")
    if W <= 0: missing.append("dimensions_in.width")
    if H <= 0: missing.append("dimensions_in.height")
    if weight_lb <= 0: missing.append("weight_lb")
    if missing:
        return {"error": f"Missing/invalid fields: {', '.join(missing)}"}

    # Volume & weights
    volume_cuft = cuft_from_inches(L, W, H) * max(pieces, 1)
    dim_w = dim_weight_lb(volume_cuft)
    chargeable = max(weight_lb, dim_w)

    # Geocode user endpoints
    o_latlng = geocode_city(o.get("city"), o.get("region"), o.get("country"))
    d_latlng = geocode_city(d.get("city"), d.get("region"), d.get("country"))
    if not o_latlng or not d_latlng:
        return {"error": "Unable to geocode origin or destination."}
    o_lat, o_lng = o_latlng
    d_lat, d_lng = d_latlng

    # Load data
    lanes = fetch_and_build_dataset(force=False)
    if lanes.empty:
        return {"error": "No historical lanes available."}

    # Step 1: radius filtering
    chosen_radius = None
    near = pd.DataFrame()
    for r in MATCH_RADII:
        mask = (
            (lanes["o_lat"].notna()) & (lanes["d_lat"].notna()) &
            (lanes.apply(lambda row: haversine_miles(o_lat, o_lng, row.o_lat, row.o_lng) <= r, axis=1)) &
            (lanes.apply(lambda row: haversine_miles(d_lat, d_lng, row.d_lat, row.d_lng) <= r, axis=1))
        )
        near = lanes[mask].copy()
        if not near.empty:
            chosen_radius = r
            break

    if near.empty:
        return {"error": "No nearby historical lanes within 100 miles. Please request a live quote."}

    # Prefer matching equipment if provided
    if equipment:
        eq = near[near["equipment"].str.lower() == str(equipment).lower()]
        if not eq.empty:
            near = eq

    # Weight banding
    has_weight_proxy = "weight_proxy_lb" in near.columns and near["weight_proxy_lb"].notna().any()
    candidates = near.copy()
    used_band = None

    if has_weight_proxy:
        for band in BANDS:
            lo, hi = (1 - band) * chargeable, (1 + band) * chargeable
            banded = near[(near["weight_proxy_lb"] >= lo) & (near["weight_proxy_lb"] <= hi)]
            if len(banded) >= 3:
                candidates = banded
                used_band = band
                break
        # fallback if banding too sparse
        if candidates.empty:
            candidates = near

    # Costs
    costs = candidates["subtotal_cad"].dropna().astype(float).tolist()
    if len(costs) == 0:
        return {"error": "No usable cost data in the matched rows."}

    costs_sorted = np.sort(costs)
    median_cost = float(np.median(costs_sorted))
    q1 = float(np.percentile(costs_sorted, 25))
    q3 = float(np.percentile(costs_sorted, 75))

    # Report distances to the *closest* lane endpoints used
    def closest_miles(lat, lng, lat_col, lng_col):
        return float(np.min(candidates.apply(lambda r: haversine_miles(lat, lng, r[lat_col], r[lng_col]), axis=1)))

    try:
        origin_mi = closest_miles(o_lat, o_lng, "o_lat", "o_lng")
        dest_mi = closest_miles(d_lat, d_lng, "d_lat", "d_lng")
    except Exception:
        origin_mi, dest_mi = None, None

    # Build response
    resp = {
        "estimate_cad": round(median_cost, 2),
        "iqr": [round(q1, 2), round(q3, 2)],
        "volume_cuft": round(volume_cuft, 2),
        "dim_weight_lb": round(dim_w, 1),
        "actual_weight_lb": round(weight_lb, 1),
        "chargeable_weight_lb": round(chargeable, 1),
        "nearby_rows_used": int(len(costs_sorted)),
        "radius_miles": int(chosen_radius or -1),
        "lane_proximity_miles": {
            "origin": round(origin_mi, 1) if origin_mi is not None else None,
            "destination": round(dest_mi, 1) if dest_mi is not None else None
        },
        "equipment_used": equipment if equipment else str(candidates["equipment"].mode().iat[0]) if not candidates["equipment"].isna().all() else None,
        "confidence": "high" if len(costs_sorted) >= 10 else ("medium" if len(costs_sorted) >= 3 else "low"),
        "notes": "Special services not included; historical subtotals may have embedded fuel/fees as originally billed."
    }

    # include a few example rows for transparency (not the full dataset)
    sample_cols = ["origin_city","origin_region","origin_country",
                   "dest_city","dest_region","dest_country",
                   "equipment","subtotal_cad"]
    if "weight_proxy_lb" in candidates.columns:
        sample_cols.append("weight_proxy_lb")

    resp["sample_rows"] = candidates[sample_cols].head(5).to_dict(orient="records")
    return resp

# ---------------------------
# Routes
# ---------------------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/refresh-cache", methods=["POST"])
def refresh_cache():
    try:
        df = fetch_and_build_dataset(force=True)
        return jsonify({"ok": True, "rows": int(len(df))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/estimate", methods=["POST"])
def estimate():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        result = compute_estimate(payload)
        status = 200 if "error" not in result else 422
        return jsonify(result), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Boot
# ---------------------------

if __name__ == "__main__":
    load_geocode_cache()
    # Warm the dataset on boot (non-fatal if it fails)
    try:
        fetch_and_build_dataset(force=False)
    except Exception as e:
        print("Warmup failed:", e)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))