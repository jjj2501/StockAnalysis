import requests
data = {
    "portfolio": [
        {
            "symbol": "AAPL",
            "shares": 50,
            "price": 247.7,
            "market": "US",
            "currency": "USD",
            "asset_type": "STOCK"
        },
        {
            "symbol": "CASH_USD",
            "shares": 100000,
            "price": 1.0,
            "market": "CASH",
            "currency": "USD",
            "asset_type": "CASH"
        }
    ],
    "force_refresh": True
}
try:
    r = requests.post("http://localhost:8000/api/portfolio/risk", json=data)
    print("STATUS:", r.status_code)
    print("TEXT:", r.text)
except Exception as e:
    print("ERROR:", e)
