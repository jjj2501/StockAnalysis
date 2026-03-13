import asyncio
import time

async def test_gpu():
    print("开始导包...")
    from backend.api.gpu_router import get_gpu_status
    from backend.core.engine import StockEngine
    print("包导入完成，准备获取 GPU 状态...")
    try:
        status = get_gpu_status()
        print(f"GPU 状态获取成功: {status}")
    except Exception as e:
        print(f"获取 GPU 状态出错: {e}")
        
    print("准备初始化 Stock Engine...")
    try:
        engine = StockEngine()
        print(f"Engine 初始化成功...")
    except Exception as e:
        print(f"初始化 Engine 出错: {e}")

if __name__ == "__main__":
    asyncio.run(test_gpu())
