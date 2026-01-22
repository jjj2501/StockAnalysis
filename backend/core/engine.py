import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from typing import List
from .model import HybridModel
from .data import DataFetcher

class StockEngine:
    def __init__(self, model_dir="backend/models", device=None):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.data_fetcher = DataFetcher()
        
        # 设备选择逻辑
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"StockEngine initialized on device: {self.device}")
        
        # 模型超参数
        self.input_dim = 12 # 对应 data.py 中的 feature_cols 数量
        self.hidden_dim = 64
        self.num_layers = 2
        self.output_dim = 1
        self.seq_length = 60
        self.learning_rate = 0.001
        self.epochs = 50 # 演示用，实际可以更多

    def _get_model(self):
        model = HybridModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        return model.to(self.device)

    def train(self, symbol: str):
        """训练指定股票的模型"""
        import time
        start_time = time.time()
        print(f"Start training for {symbol} on {self.device}...")
        df = self.data_fetcher.get_stock_data(symbol)
        df = self.data_fetcher.add_technical_indicators(df)
        X, y, scaler = self.data_fetcher.prepare_data_for_training(df, seq_length=self.seq_length)
        
        if len(X) == 0:
            print("Not enough data to train.")
            return None
            
        # 划分训练集和验证集 (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 转为 Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        # X_test_t = torch.FloatTensor(X_test).to(self.device)
        # y_test_t = torch.FloatTensor(y_test).to(self.device)

        model = self._get_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs.flatten(), y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
                
        # 保存模型
        model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
        torch.save(model.state_dict(), model_path)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Model saved to {model_path}")
        print(f"Training finished in {duration:.2f} seconds.")
        return model_path

    async def predict(self, symbol: str, task_id: str = None) -> dict:
        """预测未来趋势"""
        from .progress import progress_manager
        
        async def update_p(p, s):
            if task_id: await progress_manager.update(task_id, p, s)

        await update_p(10, "正在载入模型...")
        model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
        
        if not os.path.exists(model_path):
            await update_p(20, "模型不存在，正在训练新模型...")
            self.train(symbol)
            await update_p(50, "训练完成")
            
        model = self._get_model()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.eval()
        
        await update_p(60, "正在获取最近交易数据...")
        X_recent, df = self.data_fetcher.get_recent_data(symbol, seq_length=self.seq_length)
        
        if len(X_recent) == 0:
            return {"error": "Insufficient data"}
            
        await update_p(80, "正在执行深度学习推理...")
        X_recent_t = torch.FloatTensor(X_recent).to(self.device)
        
        with torch.no_grad():
            pred_scaled = model(X_recent_t).item()
            
        last_close = df['close'].iloc[-1]
        last_scaled_val = X_recent[0, -1, 3] 
        change_pct = (pred_scaled - last_scaled_val) / last_scaled_val if last_scaled_val != 0 else 0
        prediction_trend = "UP" if pred_scaled > last_scaled_val else "DOWN"
        
        history_df = df.tail(30)
        history_dates = history_df['date'].dt.strftime('%Y-%m-%d').tolist()
        history_prices = history_df['close'].tolist()
        
        await update_p(90, "预测计算完成")
        return {
            "symbol": symbol,
            "current_price": float(last_close),
            "predicted_trend": prediction_trend,
            "confidence": float(abs(change_pct) * 1000), 
            "model_path": model_path,
            "history_dates": history_dates,
            "history_prices": history_prices
        }
        
    def predict_batch(self, symbol: str, df: pd.DataFrame) -> List[str]:
        """
        批量预测全量趋势 (优化回测性能)
        """
        model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
        if not os.path.exists(model_path):
            self.train(symbol)
            
        model = self._get_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # 准备数据指标
        if 'MA5' not in df.columns:
            # 内部补全指标 (如果外部没传)
            df = self.data_fetcher.add_technical_indicators(df)

        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MACD', 'Signal', 'Hist', 'RSI']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        data = df[feature_cols].values
        # 归一化 (使用 DataFetcher 的 scaler 状态，这里假设已 fit)
        scaled_data = self.data_fetcher.scaler.fit_transform(data)

        # 构建滑动窗口批次
        X_batch = []
        indices = []
        for i in range(self.seq_length, len(scaled_data)):
            X_batch.append(scaled_data[i-self.seq_length:i])
            indices.append(i)

        if not X_batch:
            return ["NEUTRAL"] * len(df)

        # 转换为 Tensor 并执行批量推理
        X_t = torch.FloatTensor(np.array(X_batch)).to(self.device)
        
        predictions = []
        chunk_size = 128 # 防止显存溢出
        with torch.no_grad():
            for i in range(0, len(X_t), chunk_size):
                outputs = model(X_t[i:i+chunk_size])
                # 简单判断: 预测值 > 当前时刻最后价格对应的归一化值 则为 UP
                # 为了简化，我们直接返回预测趋势
                for j, out in enumerate(outputs):
                    last_val = X_t[i+j, -1, 3] # close index
                    predictions.append("UP" if out.item() > last_val else "DOWN")

        # 补齐开头的 seq_length 个中性结果
        return ["NEUTRAL"] * self.seq_length + predictions
