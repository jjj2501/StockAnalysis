# 缓存功能测试脚本
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.data import DataFetcher

def test_cache():
    print("=" * 50)
    print("测试缓存功能")
    print("=" * 50)
    
    # 创建数据获取器
    fetcher = DataFetcher()
    
    # 测试首次获取数据（应该从远程下载并缓存）
    print("\n【首次获取数据】")
    data = fetcher.get_stock_data("600519", "20240101", "20240131")
    print(f"获取到 {len(data)} 条记录" if len(data) > 0 else "未获取到数据")
    
    if not data.empty:
        print("数据预览:")
        print(data.tail(3))
    
    # 检查缓存文件是否存在
    cache_path = fetcher._get_cache_path("600519")
    print(f"\n缓存文件路径: {cache_path}")
    print(f"缓存文件存在: {cache_path.exists()}")
    
    if cache_path.exists():
        import os
        size = os.path.getsize(cache_path)
        print(f"缓存文件大小: {size} 字节")
    
    # 测试二次获取（应该从缓存加载）
    print("\n【二次获取数据（应从缓存加载）】")
    data2 = fetcher.get_stock_data("600519", "20240101", "20240131")
    print(f"获取到 {len(data2)} 条记录")
    
    print("\n=" * 50)
    print("测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    test_cache()
