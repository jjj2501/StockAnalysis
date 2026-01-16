
import sys
import os
import time
import torch

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.engine import StockEngine

def run_benchmark(symbol="600519", device="cpu"):
    print(f"\n{'='*20} Running Benchmark on {device.upper()} {'='*20}")
    try:
        engine = StockEngine(device=device)
        start_time = time.time()
        # 强制重新训练
        model_path = os.path.join(engine.model_dir, f"{symbol}_model.pth")
        if os.path.exists(model_path):
            os.remove(model_path)
            
        engine.train(symbol)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{device.upper()} Training Time: {duration:.4f} seconds")
        return duration
    except Exception as e:
        print(f"Error executing on {device}: {e}")
        return None

def main():
    symbol = "600519" # 贵州茅台
    
    # 1. CPU 基准测试
    cpu_time = run_benchmark(symbol, "cpu")
    
    # 2. GPU 基准测试 (如果可用)
    gpu_available = torch.cuda.is_available()
    gpu_time = None
    
    if gpu_available:
        gpu_time = run_benchmark(symbol, "cuda")
    else:
        print("\nCUDA not available, skipping GPU benchmark.")

    # 3. 结果对比
    print(f"\n{'='*20} Benchmark Results {'='*20}")
    if cpu_time:
        print(f"CPU Time: {cpu_time:.4f}s")
    
    if gpu_time:
        print(f"GPU Time: {gpu_time:.4f}s")
        if cpu_time:
            speedup = cpu_time / gpu_time
            print(f"Speedup with GPU: {speedup:.2f}x")
            diff = cpu_time - gpu_time
            print(f"Time Saved: {diff:.4f}s")
    
if __name__ == "__main__":
    main()
