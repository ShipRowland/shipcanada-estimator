"""
Configuration file for the ShipCanada estimator.

Adjust the constants here to match your historical data and desired behavior.
"""

# URL of the historical shipping data page
HISTORICAL_URL = "https://www.shipcanada.ca/shipcanada_sample_date_cost.html"

# Column name mapping from the HTML table to internal fields
# Update these values if your HTML table headers are different.
COLUMN_MAP = {
    # internal field : column name in HTML table
    "origin_city": "Origin",
    "origin_region": "Origin State/Prov",   # optional
    "origin_country": "Origin Country",     # optional
    "dest_city": "Destination",
    "dest_region": "Dest State/Prov",
    "dest_country": "Dest Country",
    "equipment": "Equipment",
    "subtotal_cad": "Sub Total",            # numeric cost column
    # if you have billed weight or similar, map it here; otherwise leave as None
    "weight_proxy_lb": None
}

# Default values when columns are missing
DEFAULT_ORIGIN_COUNTRY = "CA"
DEFAULT_DEST_COUNTRY = "CA"

# Distance radii (in miles) for matching lanes
MATCH_RADII = [50, 75, 100]

# Bands (percentage) around chargeable weight for cost matching
BANDS = [0.20, 0.40]

# Geocoder configuration
GEOCODER_USER_AGENT = "shipcanada-estimator/1.0 (operations@shipcanada.ca)"
GEOCODER_TIMEOUT = 5            # seconds
GEOCODER_SLEEP_S = 1.0          # polite delay between calls

# Cache paths
CACHE_DIR = "cache"
LANES_PARQUET = f"{CACHE_DIR}/lanes.parquet"
GEOCODE_JSON = f"{CACHE_DIR}/geocode_cache.json"

# Currency code for outputs
CURRENCY = "CAD"