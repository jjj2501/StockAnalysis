import asyncio
import logging
import time
from datetime import datetime
from backend.core.data import DataFetcher, DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

class MarketDataScheduler:
    """
    负责在后台静陌更新所有已关注标的的数据
    """
    def __init__(self):
        self.fetcher = DataFetcher()
        self.is_running = False

    async def _update_all_caches(self):
        logger.info("[Scheduler] 开始执行全局数据预热防腐更新...")
        
        if not DEFAULT_CACHE_DIR.exists():
            return
            
        # 收集所有本地已订阅的标的
        targets = []
        for file_path in DEFAULT_CACHE_DIR.glob("*.parquet"):
            filename = file_path.name.replace(".parquet", "")
            market = "CN"
            symbol = filename
            if "_" in symbol:
                parts = symbol.split("_", 1)
                market = parts[0]
                symbol = parts[1]
            targets.append((market, symbol))
            
        success_count = 0
        
        # 先拉取全局大盘宏观数据，这需要一次请求全局生效
        logger.info("[Scheduler] 正在静默更新全局宏观大盘 (CN / US)...")
        try:
            # 清除可能存在的失效旧大盘宏观缓存
            import os
            cn_macro = self.fetcher.cache_dir / "macro" / "macro_CN.json"
            us_macro = self.fetcher.cache_dir / "macro" / "macro_US.json"
            if cn_macro.exists(): os.remove(cn_macro)
            if us_macro.exists(): os.remove(us_macro)
            
            # 使用 to_thread 避免阻塞事件循环
            await asyncio.to_thread(self.fetcher.get_macro_data, "CN")
            await asyncio.to_thread(self.fetcher.get_macro_data, "US")
        except Exception as e:
            logger.error(f"[Scheduler] 宏观大盘预热更新失败: {e}")
            
        for market, symbol in targets:
            try:
                logger.info(f"[Scheduler] 正在静默更新 {market}-{symbol} 的 K线底座与多维量化因子...")
                # 强制调取从 2020 年至今的数据覆盖，内部拥有增量逻辑
                start_date = "2020-01-01"
                end_date = time.strftime("%Y-%m-%d")
                
                # 为了防止集中并发被封禁，排队串行更新K线
                df = await asyncio.to_thread(self.fetcher._fetch_from_remote, symbol, start_date, end_date, market)
                if df is not None and not df.empty:
                    await asyncio.to_thread(self.fetcher._save_cache, symbol, df, market)
                    success_count += 1
                
                # --- 静默重刷对应股票的全维因子 ---
                try:
                    factor_path = self.fetcher._get_factor_cache_path(symbol, market)
                    if factor_path.exists():
                        os.remove(factor_path) # 强制弃用旧档案
                    await asyncio.to_thread(self.fetcher.get_comprehensive_factors, f"{market}_{symbol}" if market!="CN" else symbol)
                except Exception as ef:
                    logger.warning(f"[Scheduler] 弱依赖: 静默预热全维因子失败 {market}-{symbol}: {ef}")
                
                # 喘息时间，防止被 API 提供商封锁
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"[Scheduler] 静默更新失败 {market}-{symbol}: {e}")
                
        logger.info(f"[Scheduler] 全局预热完成！共计更新 {success_count}/{len(targets)} 个标的。")

    async def start_background_loop(self):
        self.is_running = True
        logger.info("[Scheduler] 数据防腐守护进程已启动，将每日检查并自动更新！")
        
        while self.is_running:
            now = datetime.now()
            # 简化版: 每天凌晨 02:00 或 16:00 进行检查，这里采用轮询睡眠机制
            # 每 6 小时触发一次全面预热
            try:
                await self._update_all_caches()
            except Exception as e:
                logger.error(f"[Scheduler] 守护进程发生异常: {e}")
                
            # 睡 6 个小时 (6 * 3600秒)
            await asyncio.sleep(21600)
            
    def stop(self):
        self.is_running = False

# 全局单例
scheduler = MarketDataScheduler()
