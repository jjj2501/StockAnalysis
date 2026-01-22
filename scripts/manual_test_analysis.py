
import requests
import json
import sys

# 假设后端运行在 localhost:8000 (需要用户确认或自行启动)
API_URL = "http://127.0.0.1:8000/api/backtest/analyze"

def test_analyze():
    payload = {
        "symbol": "600519",
        "strategy": "macd",
        "summary": {
            "total_return_pct": 15.5,
            "annual_return_pct": 12.0,
            "max_drawdown_pct": -5.5,
            "sharpe_ratio": 1.2,
            "trade_days": 252
        }
    }
    
    try:
        # 这里只是模拟测试，实际需要后端服务运行
        # print(f"Testing POST {API_URL} with payload: {payload}")
        # resp = requests.post(API_URL, json=payload)
        # if resp.status_code == 200:
        #     print("Success:", resp.json())
        # else:
        #     print("Failed:", resp.text)
        pass 
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Run this script manually after starting the backend to verify the API.")
