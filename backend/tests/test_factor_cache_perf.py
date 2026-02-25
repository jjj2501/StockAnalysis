import time
import os
import pytest
from backend.core.data import DataFetcher

def test_factor_cache_performance_speedup():
    """
    性能测试：验证 DataFetcher 采用 JSON 日级别因子缓存后，
    第二次重复请求相同股票数据的时间是否会有断崖式的极速提升。
    """
    symbol = "600519" # 茅台
    fetcher = DataFetcher()
    
    # 清理准备：强制移除可能存在的旧缓存
    cache_path = fetcher._get_factor_cache_path(symbol, "CN")
    if cache_path.exists():
        os.remove(cache_path)
        
    # --- 第一次完全冷启动请求 (请求互联网数据库) ---
    print("\n[PERF] 第 1 次获取，模拟无日结缓存时的冷启动...")
    t1_start = time.time()
    factors1 = fetcher.get_comprehensive_factors(symbol)
    t1_cost = time.time() - t1_start
    print(f"[PERF] 第 1 次耗时: {t1_cost:.3f} 秒")
    
    # 验证是否拿到了结构化的正常响应
    assert "error" not in factors1 or factors1.get("error") == ""
    assert "fundamental" in factors1
    
    # 验证缓存文件是否已被落地
    assert cache_path.exists(), "第一次拉取后，预期 json 因子缓存文件应成功落地"
    
    # --- 第二次热启动获取 (由于已被缓存，应该以磁盘 IO 的毫秒级返回) ---
    print(f"[PERF] 第 2 次获取，测试对 {cache_path.name} 日级别缓存的断崖式提速效果...")
    t2_start = time.time()
    factors2 = fetcher.get_comprehensive_factors(symbol)
    t2_cost = time.time() - t2_start
    print(f"[PERF] 第 2 次耗时: {t2_cost:.3f} 秒")
    
    # 确保内容未受损
    assert factors1["symbol"] == factors2["symbol"]
    # 即使存在偶尔的轻微毫秒级 IO 波动，第二次也极大概率 < 1.0 秒，第一次可能需要 3s 以上，甚至是 15s+
    print(f"[PERF] 第二次仅消耗了第一次耗时的 {(t2_cost / t1_cost) * 100:.2f}%")
    assert t2_cost < 1.0, f"性能断言失败：采用缓存后，第二次加载极值不应超过 1 秒, 当前: {t2_cost:.2f}s"
    assert t2_cost < t1_cost / 2, "性能断言失败：采用缓存后，获取时间并未实现预期的 50% 以上大幅度缩减"

if __name__ == "__main__":
    # 供单脚本快速测试
    test_factor_cache_performance_speedup()
