import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from typing import List, Optional
from .model import HybridModel
from .data import DataFetcher
from .gpu_utils import get_gpu_manager
from backend.config import settings

class StockEngine:
    def __init__(self, model_dir="backend/models", device=None, use_gpu: Optional[bool] = None):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.data_fetcher = DataFetcher()
        
        # 获取GPU管理器
        self.gpu_manager = get_gpu_manager()
        
        # 设备选择逻辑
        if device:
            self.device = torch.device(device)
        else:
            # 使用GPU管理器的设备
            self.device = self.gpu_manager.get_device()
            
        print(f"StockEngine initialized on device: {self.device}")
        if self.gpu_manager.is_gpu_available():
            print(f"GPU: {self.gpu_manager.gpu_info.get('device_name', 'Unknown')}")
        
        # 模型超参数 - 从配置中读取
        self.input_dim = 12  # 对应 data.py 中的 feature_cols 数量
        self.hidden_dim = settings.MODEL_HIDDEN_DIM
        self.num_layers = settings.MODEL_NUM_LAYERS
        self.output_dim = 1
        self.seq_length = settings.SEQUENCE_LENGTH
        self.learning_rate = settings.LEARNING_RATE
        self.epochs = settings.EPOCHS
        self.batch_size = settings.BATCH_SIZE
        
        # 训练状态
        self.training_in_progress = False
        self.current_training_symbol = None

    def _get_model(self):
        model = HybridModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        # 使用GPU管理器优化模型
        model = self.gpu_manager.optimize_for_training(model, self.batch_size)
        return model

    def train(self, symbol: str, progress_callback=None):
        """训练指定股票的模型（支持GPU加速）"""
        import time
        start_time = time.time()
        
        # 检查是否已经在训练
        if self.training_in_progress:
            print(f"训练已在进行中: {self.current_training_symbol}")
            return None
            
        self.training_in_progress = True
        self.current_training_symbol = symbol
        
        try:
            print(f"开始训练 {symbol} 在设备: {self.device}")
            if self.gpu_manager.is_gpu_available():
                print(f"使用GPU加速: {self.gpu_manager.gpu_info.get('device_name', 'Unknown')}")
            
            # 获取数据
            df = self.data_fetcher.get_stock_data(symbol)
            df = self.data_fetcher.add_technical_indicators(df)
            X, y, scaler = self.data_fetcher.prepare_data_for_training(df, seq_length=self.seq_length)
            
            if len(X) == 0:
                print("数据不足，无法训练。")
                return None
                
            # 划分训练集和验证集 (80/20)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # 创建数据加载器（支持批处理）
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                pin_memory=self.gpu_manager.is_gpu_available()  # 如果使用GPU，启用pin_memory加速数据传输
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=self.batch_size * 2,  # 验证时可以使用更大的批次
                shuffle=False,
                pin_memory=self.gpu_manager.is_gpu_available()
            )

            # 初始化模型
            model = self._get_model()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # 训练循环
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            best_model_state = None
            
            print(f"开始训练，总轮次: {self.epochs}, 批次大小: {self.batch_size}")
            
            for epoch in range(self.epochs):
                # 训练阶段
                model.train()
                epoch_train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    # 移动数据到设备
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.flatten(), batch_y)
                    loss.backward()
                    
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_train_loss += loss.item() * batch_X.size(0)
                
                avg_train_loss = epoch_train_loss / len(train_loader.dataset)
                train_losses.append(avg_train_loss)
                
                # 验证阶段
                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs.flatten(), batch_y)
                        epoch_val_loss += loss.item() * batch_X.size(0)
                
                avg_val_loss = epoch_val_loss / len(val_loader.dataset)
                val_losses.append(avg_val_loss)
                
                # 更新学习率
                scheduler.step(avg_val_loss)
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                
                # 进度回调
                if progress_callback:
                    progress = (epoch + 1) / self.epochs * 100
                    progress_callback(progress, f"轮次 {epoch+1}/{self.epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
                
                # 打印进度
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"轮次 [{epoch+1}/{self.epochs}], "
                          f"训练损失: {avg_train_loss:.4f}, "
                          f"验证损失: {avg_val_loss:.4f}, "
                          f"学习率: {current_lr:.6f}")
                    
                    # 如果使用GPU，显示内存使用情况
                    if self.gpu_manager.is_gpu_available():
                        memory_info = self.gpu_manager.get_memory_info()
                        if "allocated" in memory_info:
                            print(f"GPU内存: {memory_info['allocated']:.1f} MB / {memory_info['reserved']:.1f} MB")
            
            # 保存最佳模型
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
            torch.save(best_model_state, model_path)
            
            # 保存训练历史
            history_path = os.path.join(self.model_dir, f"{symbol}_history.npz")
            np.savez(history_path, 
                    train_losses=np.array(train_losses),
                    val_losses=np.array(val_losses))
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n训练完成!")
            print(f"最佳验证损失: {best_val_loss:.4f}")
            print(f"模型保存到: {model_path}")
            print(f"训练历史保存到: {history_path}")
            print(f"总训练时间: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
            
            if self.gpu_manager.is_gpu_available():
                self.gpu_manager.clear_cache()
            
            return {
                "model_path": model_path,
                "history_path": history_path,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "training_time": duration,
                "device": str(self.device),
                "gpu_used": self.gpu_manager.is_gpu_available()
            }
            
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            self.training_in_progress = False
            self.current_training_symbol = None

    async def predict(self, symbol: str, task_id: str = None) -> dict:
        """预测未来趋势（支持GPU加速）"""
        from .progress import progress_manager
        
        async def update_p(p, s):
            if task_id: await progress_manager.update(task_id, p, s)

        await update_p(10, "正在载入模型...")
        model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
        
        if not os.path.exists(model_path):
            await update_p(20, "模型不存在，正在训练新模型...")
            
            # 定义进度回调函数
            def training_progress(progress, status):
                # 这里可以添加进度更新逻辑
                pass
                
            training_result = self.train(symbol, progress_callback=training_progress)
            if not training_result:
                return {"error": "模型训练失败"}
                
            await update_p(50, "训练完成")
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
        
    def predict_batch(self, symbol: str, df: pd.DataFrame, chunk_size: Optional[int] = None) -> List[str]:
        """
        批量预测全量趋势 (优化回测性能，支持GPU加速)
        
        Args:
            symbol: 股票代码
            df: 包含历史数据的DataFrame
            chunk_size: 分块大小，None则自动计算
            
        Returns:
            预测趋势列表
        """
        model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
        if not os.path.exists(model_path):
            print(f"模型不存在，开始训练 {symbol}...")
            self.train(symbol)
            
        model = self._get_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # 准备数据指标
        if 'MA5' not in df.columns:
            df = self.data_fetcher.add_technical_indicators(df)

        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MACD', 'Signal', 'Hist', 'RSI']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        data = df[feature_cols].values
        scaled_data = self.data_fetcher.scaler.fit_transform(data)

        # 构建滑动窗口批次
        X_batch = []
        indices = []
        for i in range(self.seq_length, len(scaled_data)):
            X_batch.append(scaled_data[i-self.seq_length:i])
            indices.append(i)

        if not X_batch:
            return ["NEUTRAL"] * len(df)

        # 转换为Tensor
        X_t = torch.FloatTensor(np.array(X_batch)).to(self.device)
        
        # 自动计算合适的chunk_size
        if chunk_size is None:
            if self.gpu_manager.is_gpu_available():
                # 根据GPU内存自动调整
                memory_info = self.gpu_manager.get_memory_info()
                if "allocated" in memory_info:
                    available_memory = self.gpu_manager.gpu_info["memory_total"] * 1024 - memory_info["reserved"]
                    # 估算每个样本所需内存
                    sample_memory = self.seq_length * len(feature_cols) * 4 / 1024**2  # MB
                    # 保留50%的安全余量
                    chunk_size = int((available_memory * 0.5) / sample_memory)
                    chunk_size = max(32, min(chunk_size, 1024))  # 限制在32-1024之间
                else:
                    chunk_size = 256
            else:
                chunk_size = 512  # CPU可以使用更大的批次
        
        print(f"批量预测: 总样本数={len(X_t)}, 分块大小={chunk_size}, 设备={self.device}")
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_t), chunk_size):
                chunk = X_t[i:i+chunk_size]
                outputs = model(chunk)
                
                # 使用向量化操作提高性能
                last_vals = chunk[:, -1, 3]  # close index
                chunk_predictions = torch.where(outputs.flatten() > last_vals, "UP", "DOWN")
                predictions.extend(chunk_predictions.cpu().numpy().tolist())
                
                # 显示进度
                if (i // chunk_size) % 10 == 0:
                    progress = (i + len(chunk)) / len(X_t) * 100
                    print(f"批量预测进度: {progress:.1f}%")

        # 补齐开头的seq_length个中性结果
        return ["NEUTRAL"] * self.seq_length + predictions
