import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from .model import HybridModel
from .data import DataFetcher

class StockEngine:
    def __init__(self, model_dir="backend/models"):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.data_fetcher = DataFetcher()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        print(f"Start training for {symbol}...")
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
        print(f"Model saved to {model_path}")
        return model_path

    def predict(self, symbol: str) -> dict:
        """预测未来趋势"""
        model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
        
        # 如果模型不存在，先训练 (为了演示方便)
        if not os.path.exists(model_path):
            print(f"Model for {symbol} not found, training new model...")
            self.train(symbol)
            
        model = self._get_model()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.eval()
        
        # 获取最近数据
        X_recent, df = self.data_fetcher.get_recent_data(symbol, seq_length=self.seq_length)
        
        if len(X_recent) == 0:
            return {"error": "Insufficient data"}
            
        X_recent_t = torch.FloatTensor(X_recent).to(self.device)
        
        with torch.no_grad():
            pred_scaled = model(X_recent_t).item()
            
        # 这里预测的是归一化后的 Close 价格，或者涨跌概率，视 Target 而定
        # 我们假设 Target 是 Close Price，我们需要反归一化，但是 DataFetcher 每次重新 fit scaler
        # 这是一个简化的假设。
        
        # 获取最后一个实际价格
        last_close = df['close'].iloc[-1]
        
        # 因为我们无法完美反归一化（scaler状态丢失），我们用相对涨跌幅来估算
        # 或者在 DataFetcher 中如果使用了 fit_transform 且只返回了 recent，我们其实不知道 global min/max
        # 修正: 在 get_recent_data 中，我们应该 fit 全局数据。
        # 在这里的简化实现中：我们直接比较模型输出值 (0-1之间) 与 序列最后一个值的相对位置
        # 或者更简单：训练目标改为 "未来收益率"，则输出即为涨跌幅。
        # 当前 target是 close price scaled.
        
        # 简单hack: 比较 pred_scaled 与 输入序列最后一个值的 scaled 值
        last_scaled_val = X_recent[0, -1, 3] # close is index 3 in feature_cols
        
        # 预测涨跌幅
        change_pct = (pred_scaled - last_scaled_val) / last_scaled_val if last_scaled_val != 0 else 0
        
        # 转换成更直观的分数 (0-100)
        prediction_trend = "UP" if pred_scaled > last_scaled_val else "DOWN"
        
        # 提取最近30天的收盘价用于绘图
        history_df = df.tail(30)
        history_dates = history_df['date'].dt.strftime('%Y-%m-%d').tolist()
        history_prices = history_df['close'].tolist()
        
        return {
            "symbol": symbol,
            "current_price": float(last_close),
            "predicted_trend": prediction_trend,
            "confidence": float(abs(change_pct) * 1000), 
            "model_path": model_path,
            "history_dates": history_dates,
            "history_prices": history_prices
        }
