import akshare as ak
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Dict
import datetime
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认缓存目录
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "market"


class DataFetcher:
    """
    数据获取与处理类
    负责从 Data Source 拉取数据, 并进行清洗、特征工程和归一化
    支持本地 Parquet 文件缓存，增量更新策略
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # 设置缓存目录
        self.cache_dir = cache_dir if cache_dir else DEFAULT_CACHE_DIR
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据缓存目录: {self.cache_dir}")
    
    def _clear_proxy(self):
        """临时清除系统代理配置，防止 akshare 连接失败"""
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
    
    def _get_cache_path(self, symbol: str) -> Path:
        """获取指定股票代码的缓存文件路径"""
        return self.cache_dir / f"{symbol}.parquet"
    
    def _load_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从本地加载缓存数据
        :param symbol: 股票代码
        :return: 缓存的 DataFrame，如不存在则返回 None
        """
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"从缓存加载 {symbol}, 共 {len(df)} 条记录")
                return df
            except Exception as e:
                logger.warning(f"读取缓存文件失败 {cache_path}: {e}")
                return None
        return None
    
    def _save_cache(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        保存数据到本地缓存
        :param symbol: 股票代码
        :param df: 要保存的 DataFrame
        :return: 是否保存成功
        """
        if df.empty:
            return False
        cache_path = self._get_cache_path(symbol)
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"已缓存 {symbol} 数据到 {cache_path}, 共 {len(df)} 条记录")
            return True
        except Exception as e:
            logger.error(f"保存缓存失败 {cache_path}: {e}")
            return False
    
    def _fetch_from_remote(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从远程 API (AkShare) 获取数据
        :param symbol: 股票代码
        :param start_date: 开始日期 YYYYMMDD
        :param end_date: 结束日期 YYYYMMDD
        :return: 清洗后的 DataFrame
        """
        try:
            # akshare stock_zh_a_hist 接口获取个股历史数据
            # adjust="qfq" 代表前复权
            self._clear_proxy()
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if df is None or df.empty:
                logger.warning(f"远程无数据: {symbol} ({start_date} - {end_date})")
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
            logger.info(f"从远程获取 {symbol} 数据: {start_date} - {end_date}, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"远程获取数据失败 {symbol}: {e}")
            return pd.DataFrame()
        
    def get_stock_data(self, symbol: str, start_date: str = "20200101", end_date: str = "20251231") -> pd.DataFrame:
        """
        获取A股历史数据（带本地缓存和增量更新）
        :param symbol: 股票代码, 例如 "600519"
        :param start_date: 开始日期 YYYYMMDD
        :param end_date: 结束日期 YYYYMMDD
        :return: 清洗后的DataFrame
        """
        # 1. 尝试加载本地缓存
        cached_df = self._load_cache(symbol)
        
        requested_start = pd.to_datetime(start_date, format='%Y%m%d')
        requested_end = pd.to_datetime(end_date, format='%Y%m%d')

        if cached_df is not None and not cached_df.empty:
            # 2. 有缓存，检查是否需要更新
            cache_min_date = cached_df['date'].min()
            cache_max_date = cached_df['date'].max()
            
            # 如果缓存能覆盖请求范围，直接返回
            if cache_min_date <= requested_start and cache_max_date >= requested_end:
                logger.info(f"{symbol} 缓存数据已覆盖请求范围 (从 {cache_min_date.strftime('%Y-%m-%d')} 到 {cache_max_date.strftime('%Y-%m-%d')})")
                return cached_df[(cached_df['date'] >= requested_start) & (cached_df['date'] <= requested_end)]

            # 如果只有结束时间落后，进行增量更新
            if cache_min_date <= requested_start and cache_max_date < requested_end:
                incremental_start = (cache_max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
                logger.info(f"{symbol} 增量更新结束日期: {incremental_start} - {end_date}")
                new_df = self._fetch_from_remote(symbol, incremental_start, end_date)
                
                if not new_df.empty:
                    combined_df = pd.concat([cached_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
                    self._save_cache(symbol, combined_df)
                    return combined_df[(combined_df['date'] >= requested_start) & (combined_df['date'] <= requested_end)]
            
            # 简化处理：如果起始时间也早于缓存，则重新拉取全部范围 (或者以后可以实现双向增量)
            logger.info(f"{symbol} 缓存范围不足 (缓存起始 {cache_min_date.strftime('%Y-%m-%d')})，重新拉取整体数据")
            df = self._fetch_from_remote(symbol, start_date, end_date)
            if not df.empty:
                # 合并缓存以防丢失其他部分
                combined_df = pd.concat([cached_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
                self._save_cache(symbol, combined_df)
                return combined_df[(combined_df['date'] >= requested_start) & (combined_df['date'] <= requested_end)]
            else:
                return cached_df[(cached_df['date'] >= requested_start) & (cached_df['date'] <= requested_end)]
        else:
            # 4. 无缓存，下载全部数据
            logger.info(f"{symbol} 无本地缓存，从远程下载数据")
            df = self._fetch_from_remote(symbol, start_date, end_date)
            if not df.empty:
                self._save_cache(symbol, df)
            return df

    def get_batch_stock_data(self, symbols: List[str], start_date: str = "20200101", end_date: str = "20251231") -> Dict[str, pd.DataFrame]:
        """
        并发获取多只股票数据 (性能优化)
        """
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
            future_to_symbol = {executor.submit(self.get_stock_data, sym, start_date, end_date): sym for sym in symbols}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"批量获取 {symbol} 失败: {e}")
                    results[symbol] = pd.DataFrame()
        return results

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

        # --- 新增加量化因子 ---
        # WR (威廉指标)
        hh = df['high'].rolling(window=14).max()
        ll = df['low'].rolling(window=14).min()
        df['WR'] = -100 * (hh - df['close']) / (hh - ll)

        # ROC (变动率指标)
        df['ROC'] = df['close'].pct_change(periods=12) * 100

        # ATR (平均真实波幅)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # BB (布林带)
        df['BB_Mid'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
        
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

    def get_comprehensive_factors(self, symbol: str, categories: Optional[List[str]] = None) -> dict:
        """
        获取综合量化因子报告 (支持分类加载与并发抓取)
        :param symbol: 股票代码
        :param categories: 因子类别列表 (None 代表获取全部)
        """
        try:
            # 默认类别
            all_cats = ['technical', 'fundamental', 'sentiment', 'northbound']
            target_cats = categories if categories else all_cats
            
            # 使用列表保存需要执行的任务
            tasks = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                if 'technical' in target_cats or 'sentiment' in target_cats:
                    tasks.append(executor.submit(self._fetch_tech_and_sentiment, symbol))
                if 'fundamental' in target_cats:
                    tasks.append(executor.submit(self._fetch_fundamental, symbol))
                if 'northbound' in target_cats:
                    tasks.append(executor.submit(self._fetch_northbound, symbol))

            results = {}
            for future in as_completed(tasks):
                res = future.result()
                results.update(res)

            # 整理返回格式
            final_report = {
                "symbol": symbol,
                "date": datetime.datetime.now().strftime('%Y-%m-%d'), 
                "technical": results.get("technical", {}),
                "sentiment": results.get("sentiment", {}),
                "fundamental": results.get("fundamental", {}),
                "northbound": results.get("northbound", {})
            }
            
            # 如果是从技术面任务获取的日期，则使用真实数据日期
            if "data_date" in results:
                final_report["date"] = results["data_date"]

            return final_report
            
        except Exception as e:
            logger.error(f"计算综合因子失败: {e}")
            return {"error": str(e)}

    def _fetch_tech_and_sentiment(self, symbol: str) -> dict:
        """获取技术面和情绪面因子"""
        try:
            _, df = self.get_recent_data(symbol, seq_length=20)
            if df.empty: return {}
            latest = df.iloc[-1]
            
            # 技术面
            tech = {
                "RSI": {"value": float(latest['RSI']), "signal": "超买" if latest['RSI'] > 70 else "超卖" if latest['RSI'] < 30 else "中性"},
                "WR": {"value": float(latest['WR']), "signal": "看涨" if latest['WR'] < -80 else "看跌" if latest['WR'] > -20 else "中性"},
                "ROC": {"value": float(latest['ROC']), "signal": "动能强" if latest['ROC'] > 0 else "动能弱"},
                "BB": {"value": float(latest['close']), "signal": "上轨压力" if latest['close'] > latest['BB_Upper'] else "下轨支撑" if latest['close'] < latest['BB_Lower'] else "轨道内"}
            }
            
            # 情绪面 (量比与换手)
            avg_vol_5d = df['volume'].iloc[-6:-1].mean() if len(df) > 6 else latest['volume']
            volume_ratio = latest['volume'] / avg_vol_5d if avg_vol_5d > 0 else 1.0
            sentiment = {
                "Turnover": {"value": float(latest.get('turnover', 0)), "unit": "%", "signal": "活跃" if latest.get('turnover', 0) > 3 else "低迷"},
                "VolumeRatio": {"value": float(volume_ratio), "signal": "放量" if volume_ratio > 1.5 else "缩量" if volume_ratio < 0.5 else "持平"}
            }
            
            return {
                "technical": tech, 
                "sentiment": sentiment, 
                "data_date": latest['date'].strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.warning(f"获取技术/情绪因子失败: {e}")
            return {}

    def _fetch_fundamental(self, symbol: str) -> dict:
        """获取基本面与成长因子 (优化: 优先使用个股实时数据)"""
        try:
            # 尝试使用个股实时行情接口 (通常比全量快)
            self._clear_proxy()
            spot_df = ak.stock_zh_a_spot_em()
            stock_spot = spot_df[spot_df['代码'] == symbol]
            if not stock_spot.empty:
                s = stock_spot.iloc[0]
                return {"fundamental": {
                    "PE": {"value": float(s['市盈率-动态']), "signal": "低估" if 0 < s['市盈率-动态'] < 15 else "高估" if s['市盈率-动态'] > 50 else "合理"},
                    "PB": {"value": float(s['市净率']), "signal": "破净" if s['市净率'] < 1 else "合理"},
                    "MarketCap": {"value": float(s['总市值'] / 1e8), "unit": "亿", "signal": "大盘股" if s['总市值'] > 1e11 else "小盘股"},
                    "ROE": {"value": 15.0, "unit": "%", "signal": "一般"} # ROE 需财报接口，暂时固定
                }}
            return {}
        except Exception as e:
            logger.warning(f"获取基本面数据失败: {e}")
            return {}

    def _fetch_northbound(self, symbol: str) -> dict:
        """获取北上资金因子"""
        try:
            self._clear_proxy()
            hsgt_df = ak.stock_hsgt_individual_em(symbol=symbol)
            if not hsgt_df.empty:
                latest_hsgt = hsgt_df.iloc[0]
                return {"northbound": {
                    "HoldingRatio": {"value": float(latest_hsgt['持股比例']), "unit": "%", "signal": "高仓位" if latest_hsgt['持股比例'] > 5 else "低仓位"},
                    "NetBuy": {"value": float(latest_hsgt['当日增持市值'] / 1e4), "unit": "万", "signal": "连续买入" if latest_hsgt['当日增持市值'] > 0 else "资金流出"}
                }}
            return {}
        except Exception as e:
            logger.warning(f"获取北上资金数据失败: {e}")
            return {}
