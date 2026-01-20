import sys
import os
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.data import DataFetcher
from backend.core.backtester import BacktestEngine, MACDStrategy, RSIStrategy

def test_backtest_logic():
    print(">>> 启动回测逻辑验证测试...")
    symbol = "600519"
    fetcher = DataFetcher()
    
    # 获取数据 (2024年)
    print(f"正在获取 {symbol} 数据...")
    df = fetcher.get_stock_data(symbol, start_date="20240101", end_date="20251231")
    df = fetcher.add_technical_indicators(df)
    
    print(f"获取到数据起始日期: {df['date'].min()}, 结束日期: {df['date'].max()}")
    
    # 截取 2024 年
    df_test = df[(df['date'] >= pd.to_datetime("2024-01-01")) & (df['date'] <= pd.to_datetime("2024-12-31"))]
    print(f"筛选出的 2024 年回测数据条数: {len(df_test)}")
    
    backtester = BacktestEngine(initial_cash=1000000.0)
    
    # 1. 测试 MACD 策略
    print("\n[测试 MACD 策略]")
    res_macd = backtester.run(df_test, MACDStrategy())
    summary = res_macd['summary']
    print(f"最终资产: {summary['final_equity']:.2f}")
    print(f"总收益率: {summary['total_return_pct']:.2f}%")
    print(f"交易笔数: {len(res_macd['trades'])}")
    
    # 2. 测试 RSI 策略
    print("\n[测试 RSI 策略]")
    res_rsi = backtester.run(df_test, RSIStrategy())
    summary_rsi = res_rsi['summary']
    print(f"最终资产: {summary_rsi['final_equity']:.2f}")
    print(f"总收益率: {summary_rsi['total_return_pct']:.2f}%")
    print(f"交易笔数: {len(res_rsi['trades'])}")
    
    print("\n>>> 验证测试完成!")

if __name__ == "__main__":
    test_backtest_logic()
