import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import datetime

class DataFetcher:
    """
    数据获取与处理类
    负责从 Data Source 拉取数据, 并进行清洗、特征工程和归一化
    """
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def get_stock_data(self, symbol: str, start_date: str = "20200101", end_date: str = "20251231") -> pd.DataFrame:
        """
        获取A股历史数据
        :param symbol: 股票代码, 例如 "600519"
        :param start_date: 开始日期 YYYYMMDD
        :param end_date: 结束日期 YYYYMMDD
        :return: 清洗后的DataFrame
        """
        try:
            # akshare stock_zh_a_hist 接口获取个股历史数据
            # adjust="qfq" 代表前复权
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if df is None or df.empty:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()
            
            # 重命名列以方便使用
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_chg",
                "涨跌额": "change",
                "换手率": "turnover"
            })
            
            # 确保日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        if df.empty:
            return df
            
        # 简单移动平均 (Simple Moving Averages)
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 填充NaN (前向后向填充)
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df

    def prepare_data_for_training(self, df: pd.DataFrame, target_col: str = 'close', seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        准备训练数据 (序列化 + 归一化)
        :param df: 数据表
        :param target_col: 预测目标列
        :param seq_length: 时间窗口长度
        :return: X, y, scaler
        """
        if df.empty:
            return np.array([]), np.array([]), self.scaler

        # 选择特征列
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MACD', 'Signal', 'Hist', 'RSI']
        # 确保列存在
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        data = df[feature_cols].values
        
        # 归一化
        scaled_data = self.scaler.fit_transform(data)
        
        X = []
        y = []
        
        # 获取目标列的索引
        target_idx = feature_cols.index(target_col)
        
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i-seq_length:i])
            # 预测下一个时间点的目标值 (可以改为预测未来收益率)
            y.append(scaled_data[i, target_idx])
            
        return np.array(X), np.array(y), self.scaler

    def get_recent_data(self, symbol: str, seq_length: int = 60) -> Tuple[np.ndarray, pd.DataFrame]:
        """为预测获取最近的数据"""
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        # 获取足够长的数据用于计算指标
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        
        df = self.get_stock_data(symbol, start_date=start_date, end_date=end_date)
        df = self.add_technical_indicators(df)
        
        if len(df) < seq_length:
            # 数据不足
            return np.array([]), df
            
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MACD', 'Signal', 'Hist', 'RSI']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        data = df[feature_cols].values
        # 注意：这里重新 fit 可能会导致与训练时分布不一致。
        # 理想情况下应该加载训练好的 scaler。但在简单版本中，我们假设每个股票分布独立归一化。
        scaled_data = self.scaler.fit_transform(data)
        
        # 取最后 seq_length 个作为输入
        last_sequence = scaled_data[-seq_length:]
        return np.expand_dims(last_sequence, axis=0), df
