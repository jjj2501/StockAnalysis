import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any
from backend.core.data import DataFetcher

logger = logging.getLogger(__name__)

# 数据管理目录与配置文件 (位于 backend/data 下)
MONITORED_STOCKS_FILE = Path(__file__).parent.parent / "data" / "monitored_stocks.json"

class DataManager:
    """
    市场数据中心化管理逻辑
    负责管理需要同步的股票池，并执行批量抓取任务
    """
    def __init__(self):
        self.fetcher = DataFetcher()
        self.config_path = MONITORED_STOCKS_FILE
        # 确保目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self._save_config([])

    def _load_config(self) -> List[str]:
        """加载监控股票列表"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载监控配置失败: {e}")
            return []

    def _save_config(self, stocks: List[str]):
        """保存监控股票列表"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(stocks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存监控配置失败: {e}")

    def get_monitored_stocks(self) -> List[Dict[str, Any]]:
        """获取带状态的监控列表"""
        symbols = self._load_config()
        results = []
        for symbol in symbols:
            # 检查本地缓存状态
            cache_path = self.fetcher._get_cache_path(symbol)
            status = "未同步"
            last_date = "--"
            
            if cache_path.exists():
                df = self.fetcher._load_cache(symbol)
                if df is not None and not df.empty:
                    last_date = df['date'].max().strftime('%Y-%m-%d')
                    status = "已缓存"
            
            results.append({
                "symbol": symbol,
                "status": status,
                "last_sync": last_date
            })
        return results

    def add_stock(self, symbol: str) -> bool:
        """添加监控股票"""
        stocks = self._load_config()
        if symbol not in stocks:
            stocks.append(symbol)
            self._save_config(stocks)
            return True
        return False

    def remove_stock(self, symbol: str) -> bool:
        """移除监控股票"""
        stocks = self._load_config()
        if symbol in stocks:
            stocks.remove(symbol)
            self._save_config(stocks)
            return True
        return False

    def sync_stock(self, symbol: str) -> bool:
        """同步单只股票数据"""
        try:
            # 使用 DataFetcher 的 get_stock_data 逻辑进行增量更新
            # 默认拉取所有可用数据到今天
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            df = self.fetcher.get_stock_data(symbol, end_date=end_date)
            return not df.empty
        except Exception as e:
            logger.error(f"同步股票 {symbol} 失败: {e}")
            return False

    def sync_all(self) -> Dict[str, bool]:
        """批量同步所有股票"""
        symbols = self._load_config()
        sync_results = {}
        for symbol in symbols:
            success = self.sync_stock(symbol)
            sync_results[symbol] = success
        return sync_results
