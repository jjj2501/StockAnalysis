import sys
import logging

logging.basicConfig(level=logging.INFO)

from backend.core.risk import RiskManager

portfolio = [
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
]

rm = RiskManager()
res = rm.calculate_portfolio_risk(portfolio, days_history=100)
print("FINAL RESULT:", res)
