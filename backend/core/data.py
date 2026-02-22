import akshare as ak
import yfinance as yf
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
        """临时清除系统代理配置，彻底防止 akshare 连接失败"""
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
            # 强制清空值
            os.environ[var] = ''
            
        # 强制 requests 不走任何代理
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'
    
    def _get_cache_path(self, symbol: str, market: str = "CN") -> Path:
        """获取指定股票代码的缓存文件路径"""
        return self.cache_dir / f"{market}_{symbol}.parquet"
    
    def _load_cache(self, symbol: str, market: str = "CN") -> Optional[pd.DataFrame]:
        """
        从本地加载缓存数据
        :param symbol: 代码
        :param market: 市场
        :return: 缓存的 DataFrame，如不存在则返回 None
        """
        cache_path = self._get_cache_path(symbol, market)
        # 兼容旧格式缓存文件（不含 market 前缀，如 600519.parquet）
        legacy_cache_path = self.cache_dir / f"{symbol}.parquet"
        
        # 优先加载新格式缓存
        for path in [cache_path, legacy_cache_path]:
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    df['date'] = pd.to_datetime(df['date'])
                    logger.info(f"从缓存加载 {market}-{symbol} ({path.name}), 共 {len(df)} 条记录")
                    return df
                except Exception as e:
                    logger.warning(f"读取缓存文件失败 {path}: {e}")
                    continue
        return None
    
    def _save_cache(self, symbol: str, df: pd.DataFrame, market: str = "CN") -> bool:
        """
        保存数据到本地缓存
        :param symbol: 代码
        :param df: 要保存的 DataFrame
        :param market: 市场
        :return: 是否保存成功
        """
        if df.empty:
            return False
        cache_path = self._get_cache_path(symbol, market)
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"已缓存 {market}-{symbol} 数据到 {cache_path}, 共 {len(df)} 条记录")
            return True
        except Exception as e:
            logger.error(f"保存缓存失败 {cache_path}: {e}")
            return False
    
    def _fetch_from_remote(self, symbol: str, start_date: str, end_date: str, market: str = "CN") -> pd.DataFrame:
        """从远程 API 获取数据 (按市场路由策略)"""
        if market == "CN":
            return self._fetch_from_akshare_a(symbol, start_date, end_date)
        elif market in ["US", "HK", "CRYPTO", "FOREX", "INDEX"]:
            return self._fetch_from_yfinance(symbol, start_date, end_date, market)
        elif market == "CASH":
            return self._fetch_mock_cash(start_date, end_date)
        else:
            logger.error(f"不支持的市场类型: {market}")
            return pd.DataFrame()

    def _fetch_from_akshare_a(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            self._clear_proxy()
            # 使用线程池实现超时控制，防止 akshare 网络卡死
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ak.stock_zh_a_hist,
                    symbol=symbol, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq"
                )
                try:
                    df = future.result(timeout=15)  # 15 秒超时
                except Exception as timeout_err:
                    logger.warning(f"AkShare 请求超时（15s）{symbol}: {timeout_err}")
                    return pd.DataFrame()
            
            if df is None or df.empty:
                logger.warning(f"远程无数据: {symbol} ({start_date} - {end_date})")
                return pd.DataFrame()
            
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover"
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            logger.info(f"从 AkShare 获取 {symbol} 数据: {start_date} - {end_date}, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"AkShare 获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_yfinance(self, symbol: str, start_date: str, end_date: str, market: str) -> pd.DataFrame:
        try:
            start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            yf_symbol = symbol
            if market == "HK" and not symbol.endswith(".HK"):
                yf_symbol = f"{symbol}.HK"
                
            logger.info(f"从 yfinance 获取 {yf_symbol} ({market}): {start_dt} - {end_dt}")
            self._clear_proxy()
            df = yf.download(yf_symbol, start=start_dt, end=end_dt, progress=False)
            
            if df is None or df.empty:
                logger.warning(f"yfinance 无数据: {yf_symbol}")
                return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            # 兼容列名
            rename_map = {"Date": "date", "Datetime": "date", "Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume", "Adj Close": "adj_close"}
            df = df.rename(columns=rename_map)
            df.columns = [c.lower() for c in df.columns]
            
            if 'adj_close' in df.columns:
                df['close'] = df['adj_close']
            
            df['amount'] = df['close'] * df['volume']
            df['amplitude'] = (df['high'] - df['low']) / df['open'].replace(0, pd.NA).fillna(1) * 100
            df['change'] = df['close'].diff()
            df['pct_chg'] = df['close'].pct_change() * 100
            df['turnover'] = 0.0
            
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.sort_values('date')
            df.fillna(method='ffill', inplace=True)
            return df
        except Exception as e:
            logger.error(f"yfinance 获取数据失败 {symbol} ({market}): {e}")
            return pd.DataFrame()

    def _fetch_mock_cash(self, start_date: str, end_date: str) -> pd.DataFrame:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        df = pd.DataFrame({'date': dates})
        df['open'] = 1.0
        df['close'] = 1.0
        df['high'] = 1.0
        df['low'] = 1.0
        df['volume'] = 0.0
        df['amount'] = 0.0
        df['amplitude'] = 0.0
        df['change'] = 0.0
        df['pct_chg'] = 0.0
        df['turnover'] = 0.0
        return df
        
    def get_stock_data(self, symbol: str, start_date: str = "20200101", end_date: str = "20251231", market: str = "CN") -> pd.DataFrame:
        """
        获取历史数据（带本地缓存和增量更新，支持多市场路由）
        :param symbol: 资产代码
        :param market: 市场标识 ("CN", "US", "HK", "CRYPTO", "CASH")
        """
        cached_df = self._load_cache(symbol, market)
        
        requested_start = pd.to_datetime(start_date, format='%Y%m%d')
        requested_end = pd.to_datetime(end_date, format='%Y%m%d')

        if cached_df is not None and not cached_df.empty:
            cache_min_date = cached_df['date'].min()
            cache_max_date = cached_df['date'].max()
            
            if cache_min_date <= requested_start and cache_max_date >= requested_end:
                logger.info(f"{market}-{symbol} 缓存数据已覆盖请求范围")
                return cached_df[(cached_df['date'] >= requested_start) & (cached_df['date'] <= requested_end)]

            if cache_min_date <= requested_start and cache_max_date < requested_end:
                incremental_start = (cache_max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
                logger.info(f"{market}-{symbol} 增量更新结束日期: {incremental_start} - {end_date}")
                new_df = self._fetch_from_remote(symbol, incremental_start, end_date, market)
                
                if not new_df.empty:
                    combined_df = pd.concat([cached_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
                    self._save_cache(symbol, combined_df, market)
                    return combined_df[(combined_df['date'] >= requested_start) & (combined_df['date'] <= requested_end)]
                else:
                    # 远程获取失败时，使用已有缓存数据（可能不完整但比空结果好）
                    logger.warning(f"{market}-{symbol} 增量更新失败，使用已有缓存数据")
                    return cached_df[(cached_df['date'] >= requested_start) & (cached_df['date'] <= requested_end)]
            
            logger.info(f"{market}-{symbol} 缓存范围不足，重新拉取整体数据")
            df = self._fetch_from_remote(symbol, start_date, end_date, market)
            if not df.empty:
                combined_df = pd.concat([cached_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
                self._save_cache(symbol, combined_df, market)
                return combined_df[(combined_df['date'] >= requested_start) & (combined_df['date'] <= requested_end)]
            else:
                return cached_df[(cached_df['date'] >= requested_start) & (cached_df['date'] <= requested_end)]
        else:
            logger.info(f"{market}-{symbol} 无本地缓存，从远程下载数据")
            df = self._fetch_from_remote(symbol, start_date, end_date, market)
            if not df.empty:
                self._save_cache(symbol, df, market)
            return df

    def get_batch_stock_data(self, symbols: List[str], start_date: str = "20200101", end_date: str = "20251231", market: str = "CN") -> Dict[str, pd.DataFrame]:
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
            all_cats = ['technical', 'fundamental', 'sentiment', 'northbound', 'news']
            target_cats = categories if categories else all_cats
            
            # 使用列表保存需要执行的任务
            tasks = []
            with ThreadPoolExecutor(max_workers=6) as executor:
                if 'technical' in target_cats or 'sentiment' in target_cats:
                    tasks.append(executor.submit(self._fetch_tech_and_sentiment, symbol))
                if 'fundamental' in target_cats:
                    tasks.append(executor.submit(self._fetch_fundamental, symbol))
                if 'northbound' in target_cats:
                    tasks.append(executor.submit(self._fetch_northbound, symbol))
                if 'news' in target_cats:
                    tasks.append(executor.submit(self._fetch_news, symbol))

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
                "northbound": results.get("northbound", {}),
                "news": results.get("news", {})
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
        """获取基本面因子（使用个股信息接口获取基础数据）"""
        try:
            self._clear_proxy()
            # 使用超时控制获取个股基础信息
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ak.stock_individual_info_em, symbol=symbol)
                try:
                    info_df = future.result(timeout=10)
                except Exception as timeout_err:
                    logger.warning(f"基本面数据请求超时（10s）{symbol}: {timeout_err}")
                    return {}
            
            if info_df is None or info_df.empty:
                logger.warning(f"基本面数据为空: {symbol}")
                return {}
            
            # 转为字典
            info_dict = dict(zip(info_df.iloc[:, 0], info_df.iloc[:, 1]))
            logger.info(f"基本面字段: {list(info_dict.keys())}")
            
            # 提取可用指标
            market_cap_raw = self._safe_float(info_dict, ['总市值'], 0)
            market_cap = market_cap_raw / 1e8 if market_cap_raw > 1e6 else market_cap_raw
            flow_cap_raw = self._safe_float(info_dict, ['流通市值'], 0)
            flow_cap = flow_cap_raw / 1e8 if flow_cap_raw > 1e6 else flow_cap_raw
            latest_price = self._safe_float(info_dict, ['最新'], 0)
            total_shares = self._safe_float(info_dict, ['总股本'], 0)
            
            # 从最新价和总股本反推简易估值（总市值/净利润近似，此处仅提供市值数据）
            return {"fundamental": {
                "PE": {"value": 0, "signal": "暂无数据"},  # PE 需要净利润数据，当前接口无法提供
                "PB": {"value": 0, "signal": "暂无数据"},  # PB 需要净资产数据
                "MarketCap": {"value": round(market_cap, 2), "unit": "亿", "signal": "大盘股" if market_cap_raw > 1e11 else "小盘股"},
                "FlowCap": {"value": round(flow_cap, 2), "unit": "亿", "signal": ""},
                "LatestPrice": {"value": latest_price, "unit": "元", "signal": ""},
                "Industry": {"value": str(info_dict.get('行业', '未知')), "signal": ""}
            }}
        except Exception as e:
            logger.warning(f"获取基本面数据失败: {e}")
            return {}

    def _fetch_northbound(self, symbol: str) -> dict:
        """获取北上资金因子（自动适配列名变更）"""
        try:
            self._clear_proxy()
            # 使用线程池超时控制
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ak.stock_hsgt_individual_em, symbol=symbol)
                try:
                    hsgt_df = future.result(timeout=20)  # 20 秒超时
                except Exception as timeout_err:
                    logger.warning(f"北上资金数据请求超时（20s）{symbol}: {timeout_err}")
                    return {}
            
            if hsgt_df is None or hsgt_df.empty:
                logger.warning(f"北上资金数据为空: {symbol}")
                return {}
            
            # 记录实际列名（便于调试 akshare 版本变更）
            logger.info(f"北上资金数据列名: {list(hsgt_df.columns)}")
            latest = hsgt_df.iloc[0]
            
            # 容错列名匹配：适配不同 akshare 版本（基于实际日志确认的列名）
            holding_ratio = self._safe_col_float(latest, ['持股数量占A股百分比', '持股比例', '持股比例(%)', '比例', '占比'], 0)
            net_buy = self._safe_col_float(latest, ['今日增持资金', '当日增持市值', '当日增持估计净买额', '增持市值', '净买额'], 0)
            
            return {"northbound": {
                "HoldingRatio": {"value": holding_ratio, "unit": "%", "signal": "高仓位" if holding_ratio > 5 else "低仓位"},
                "NetBuy": {"value": round(net_buy / 1e4, 2) if abs(net_buy) > 1e4 else net_buy, "unit": "万", "signal": "连续买入" if net_buy > 0 else "资金流出"}
            }}
        except Exception as e:
            logger.warning(f"获取北上资金数据失败: {e}")
            return {}

    def _fetch_news(self, symbol: str) -> dict:
        """获取个股新闻因子（最新 10 条新闻 + 情感摘要）"""
        try:
            self._clear_proxy()
            # 使用超时控制获取个股新闻
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ak.stock_news_em, symbol=symbol)
                try:
                    news_df = future.result(timeout=15)  # 15 秒超时
                except Exception as timeout_err:
                    logger.warning(f"新闻数据请求超时（15s）{symbol}: {timeout_err}")
                    return {}
            
            if news_df is None or news_df.empty:
                logger.warning(f"新闻数据为空: {symbol}")
                return {}
            
            # 记录实际列名（便于调试 akshare 版本变更）
            logger.info(f"新闻数据列名: {list(news_df.columns)}")
            
            # 取最新 10 条新闻
            news_list = []
            for _, row in news_df.head(10).iterrows():
                # 容错列名（适配不同 akshare 版本）
                title = ''
                for col in ['新闻标题', '标题', 'title']:
                    if col in row.index and row[col]:
                        title = str(row[col])
                        break
                
                pub_time = ''
                for col in ['发布时间', '时间', '日期', 'datetime', 'date']:
                    if col in row.index and row[col]:
                        pub_time = str(row[col])
                        break
                
                source = ''
                for col in ['文章来源', '来源', 'source']:
                    if col in row.index and row[col]:
                        source = str(row[col])
                        break
                
                url = ''
                for col in ['新闻链接', '链接', 'url', 'link']:
                    if col in row.index and row[col]:
                        url = str(row[col])
                        break
                
                if title:  # 只添加有标题的新闻
                    news_list.append({
                        "title": title,
                        "time": pub_time,
                        "source": source,
                        "url": url
                    })
            
            # 简单情感分析：基于标题关键词统计
            positive_words = ['涨', '突破', '创新高', '大涨', '增长', '利好', '超预期', '强势', '反弹', '上涨', '买入', '看多']
            negative_words = ['跌', '下跌', '大跌', '暴跌', '下滑', '利空', '风险', '警告', '发行', '清仓', '看空', '减持']
            
            pos_count = 0
            neg_count = 0
            for item in news_list:
                t = item['title']
                pos_count += sum(1 for w in positive_words if w in t)
                neg_count += sum(1 for w in negative_words if w in t)
            
            total = pos_count + neg_count
            if total > 0:
                sentiment_score = round((pos_count - neg_count) / total * 100, 1)
            else:
                sentiment_score = 0
            
            if sentiment_score > 20:
                sentiment_signal = "偏多"
            elif sentiment_score < -20:
                sentiment_signal = "偏空"
            else:
                sentiment_signal = "中性"
            
            logger.info(f"新闻因子: {len(news_list)} 条新闻, 情感分={sentiment_score}, 信号={sentiment_signal}")
            
            return {"news": {
                "items": news_list,
                "sentiment_score": sentiment_score,
                "sentiment_signal": sentiment_signal,
                "count": len(news_list)
            }}
        except Exception as e:
            logger.warning(f"获取新闻数据失败: {e}")
            return {}

    @staticmethod
    def _safe_float(data: dict, keys: list, default: float = 0) -> float:
        """从字典中安全提取浮点数值，尝试多个候选键名"""
        for key in keys:
            if key in data:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    continue
        return default

    @staticmethod
    def _safe_col_float(row, keys: list, default: float = 0) -> float:
        """从 DataFrame 行中安全提取浮点数值，尝试多个候选列名"""
        for key in keys:
            if key in row.index:
                try:
                    return float(row[key])
                except (ValueError, TypeError):
                    continue
        return default
