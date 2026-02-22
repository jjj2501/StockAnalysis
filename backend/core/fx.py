import yfinance as yf
import pandas as pd
import logging
import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 默认缓存目录
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "fx"

class FXManager:
    """
    外汇汇率管理模块
    负责获取和缓存多币种对基础结算货币（如 CNY）的历史每日汇率折线
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, from_curr: str, to_curr: str) -> Path:
        return self.cache_dir / f"{from_curr}_{to_curr}.parquet"

    def get_fx_rate_series(self, from_curr: str, to_curr: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取每日汇率序列 (from_curr -> to_curr)
        如果二者相同，则返回常数 1.0 的平滑折线
        """
        from_curr = from_curr.upper()
        to_curr = to_curr.upper()
        
        # 转换日期格式 (20250101 -> 2025-01-01)
        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)
        
        # 1. 如果币种相同，直接创建全是 1.0 的序列
        if from_curr == to_curr:
            dates = pd.date_range(start=req_start, end=req_end, freq='B')
            return pd.Series(1.0, index=dates)

        # yfinance 汇率 symbol 格式: "USDCNY=X"
        symbol = f"{from_curr}{to_curr}=X"
        cache_path = self._get_cache_path(from_curr, to_curr)
        
        # 检查缓存
        cached_df = None
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                cached_df['date'] = pd.to_datetime(cached_df['date'])
                cached_df = cached_df.set_index('date')
            except Exception as e:
                logger.warning(f"读取汇率缓存失败 {cache_path}: {e}")
        
        # TODO: 为简化逻辑，每次直接向 yfinance 请求或做粗略缓存拼接。
        # 这里先实现全量或增量下载后合并的方法
        cache_min = cached_df.index.min() if cached_df is not None and not cached_df.empty else None
        cache_max = cached_df.index.max() if cached_df is not None and not cached_df.empty else None
        
        # 需要远程下载的标记
        need_download = True
        if cached_df is not None and not cached_df.empty:
            if cache_min <= req_start and cache_max >= req_end:
                need_download = False
                
        if need_download:
            logger.info(f"从 yfinance 获取汇率: {symbol} ({req_start.strftime('%Y-%m-%d')} - {req_end.strftime('%Y-%m-%d')})")
            try:
                dl_start = req_start.strftime('%Y-%m-%d')
                dl_end = (req_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                df = yf.download(symbol, start=dl_start, end=dl_end, progress=False)
                
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    # 保留 Close 作为基准汇率
                    new_df = pd.DataFrame({'rate': df['Close']})
                    new_df.index = pd.to_datetime(new_df.index).tz_localize(None)
                    
                    # 合并缓存
                    if cached_df is not None and not cached_df.empty:
                        combined = pd.concat([cached_df, new_df])
                        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                    else:
                        combined = new_df
                        
                    # 保存回缓存
                    try:
                        combined_reset = combined.reset_index(names=['date'])
                        combined_reset.to_parquet(cache_path, index=False)
                    except Exception as e:
                        logger.error(f"保存汇率缓存失败 {cache_path}: {e}")
                        
                    cached_df = combined
            except Exception as e:
                logger.error(f"yfinance 获取汇率失败 {symbol}: {e}")
        
        if cached_df is not None and not cached_df.empty:
            result = cached_df.copy()
            # 补齐节假日等缺失的日期 (前向填充)
            full_idx = pd.date_range(start=req_start, end=req_end, freq='B')
            result = result.reindex(full_idx)
            result['rate'] = result['rate'].fillna(method='ffill').fillna(method='bfill')
            return result['rate']
            
        # 兜底：如果实在查不到汇率，返回1.0（虽然不准确，但防崩溃）
        logger.warning(f"无法获取有效汇率 {symbol}，使用 1.0 替代")
        dates = pd.date_range(start=req_start, end=req_end, freq='B')
        return pd.Series(1.0, index=dates)
