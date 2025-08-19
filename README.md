# ShipCanada Estimator API

A tiny Flask API that converts shipment inputs into a **ballpark estimate** using ShipCanada historical data.

## One‑time setup

1. **Clone or download this repository**

   ```bash
   git clone https://github.com/<your-user>/shipcanada-estimator.git
   cd shipcanada-estimator
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows use .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Run the API locally**

   ```bash
   python app.py
   ```

   The server runs on `http://localhost:8000` by default.

4. **(Optional) Pre‑build the cache**

   ```bash
   curl -X POST http://localhost:8000/refresh-cache
   ```

   This scrapes `HISTORICAL_URL`, normalizes columns per `config.py`, geocodes cities (with caching in `cache/geocode_cache.json`), and writes a parquet dataset (`cache/lanes.parquet`).

## Estimate example

Send a POST request to `/estimate` with JSON:

```bash
curl -s -X POST http://localhost:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "origin": {"city":"Toronto","region":"ON","country":"CA"},
    "destination": {"city":"Vancouver","region":"BC","country":"CA"},
    "dimensions_in": {"length":48,"width":48,"height":48},
    "weight_lb": 600,
    "pieces": 1,
    "equipment": "Van"
  }' | jq .
```

You will receive a JSON response with an estimated cost, quartiles, and additional details.

## Deploying to Render

1. Sign in to [Render](https://render.com/) and create a **new Web Service**.
2. Connect this repository from GitHub.
3. Set the **Build Command** to:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the **Start Command** to:
   ```bash
   gunicorn app:app --preload --workers=2 --bind 0.0.0.0:$PORT
   ```
5. Once deployed, POST to `https://<your-service>.onrender.com/refresh-cache` to build the data cache.

## Connect to Chatbase

In Chatbase, create an **Action** pointing to your `/estimate` endpoint. Map user inputs (pickup city, destination, dimensions, weight, pieces, equipment) to the JSON body. Display the response fields in the chat as described in the instructions.