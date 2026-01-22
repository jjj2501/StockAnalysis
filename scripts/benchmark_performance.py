import sys
import os
import time
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.data import DataFetcher
from backend.core.backtester import BacktestEngine, MACDStrategy, RSIStrategy, AIStrategy
from backend.core.engine import StockEngine

def benchmark():
    print(">>> 启动性能基准测试...")
    symbol = "600519"
    fetcher = DataFetcher()
    engine = StockEngine()
    
    # 1. 准备数据
    df = fetcher.get_stock_data(symbol, start_date="20240101", end_date="20250101")
    df = fetcher.add_technical_indicators(df)
    
    backtester = BacktestEngine(initial_cash=1000000.0)
    
    # 2. 测试向量化回测速度 (MACD)
    print("\n[测试 MACD 向量化回测]")
    t1 = time.time()
    res_vec = backtester.run_vectorized(df, MACDStrategy())
    t2 = time.time()
    print(f"向量化回测耗时: {(t2-t1)*1000:.2f} ms")
    print(f"结果模式: {res_vec.get('mode')}")
    
    # 3. 测试 AI 批量推理速度
    print("\n[测试 AI 批量推理回测]")
    ai_strategy = AIStrategy(engine, symbol)
    t3 = time.time()
    # 内部会调用 predict_batch
    res_ai = backtester.run_vectorized(df, ai_strategy)
    t4 = time.time()
    print(f"AI 批量推理+向量化回测总耗时: {(t4-t3)*1000:.2f} ms")
    print(f"回测期间天数: {len(df)}")
    
    print("\n>>> 性能优化验证完成!")

if __name__ == "__main__":
    benchmark()
