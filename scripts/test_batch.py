# 批量并发获取测试脚本
import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.data import DataFetcher

def test_batch():
    fetcher = DataFetcher()
    # 选择几支具有代表性的股票
    symbols = ["600519", "000001", "000002", "600000"]
    
    print(f"开始并行获取 {len(symbols)} 支股票的数据...")
    start_time = time.time()
    
    results = fetcher.get_batch_data(symbols, start_date="20240101", end_date="20240110", max_workers=len(symbols))
    
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    for symbol, df in results.items():
        print(f"股票 {symbol}: 获取到 {len(df)} 条记录")

if __name__ == "__main__":
    test_batch()
