import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.engine import StockEngine

def main():
    parser = argparse.ArgumentParser(description="手动训练股票模型")
    parser.add_argument("symbol", type=str, help="股票代码 (例如: 600519)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    args = parser.parse_args()

    engine = StockEngine()
    # 覆盖默认训练轮数
    engine.epochs = args.epochs
    
    print(f"正在为 {args.symbol} 开始训练 (Epochs: {args.epochs})...")
    try:
        model_path = engine.train(args.symbol)
        if model_path:
            print(f"训练成功! 模型已保存至: {model_path}")
        else:
            print("训练失败 (可能是因为数据不足)")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
