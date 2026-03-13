import akshare as ak
import sys
from pathlib import Path

# 将项目根目录加入环境变量以便引包
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.rag.ingestion import KnowledgeIngestion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_real_ingestion(symbol: str = "600519"):
    """
    通过 Akshare 拉取指定沪深股票的最新东方财富新闻资讯
    拼接为长文后，切割送入本地 ChromaDB。
    """
    logger.info(f"开始拉取 {symbol} 的真实市场新闻/公告材料...")
    
    try:
        # 获取个股新闻 (东方财富源)
        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is None or news_df.empty:
            logger.warning(f"未能拉取到 {symbol} 的新闻数据。")
            return
            
        # 提取最近的 30 条新闻标题与内容拼接成一篇“近期基本面总集”
        recent_news = news_df.head(30)
        
        long_text_parts = [f"【AlphaPulse V0.4 基本面资料集 - {symbol} 近期动态全景】\n"]
        
        for index, row in recent_news.iterrows():
            title = row.get("新闻标题", "")
            content = row.get("新闻内容", "")
            publish_time = row.get("发布时间", "近期")
            
            # 部分新闻只有标题没有清洗好的正文，在此作 fallback
            if pd.isna(content) or not content:
                content = title
                
            article = f"[{publish_time}] {title}\n摘要/内容: {content}\n"
            long_text_parts.append(article)
            
        full_document = "\n".join(long_text_parts)
        
        logger.info(f"成功拼接长文本，共 {len(full_document)} 字符，准备切片入库...")
        
        # 调取我们的 RAG 知识导入管线
        pipeline = KnowledgeIngestion(chunk_size=600, chunk_overlap=100)
        success = pipeline.process_and_store_document(
            symbol=symbol,
            source_type="news_and_announcements",
            raw_text=full_document
        )
        
        if success:
            logger.info("✅ 真实数据成功切片并硬写入 ChromaDB！实弹打流测试完成！")
        else:
            logger.error("❌ 写入向量库失败！")
            
    except Exception as e:
        logger.error(f"拉取或处理数据时发生崩溃: {e}")

if __name__ == "__main__":
    import pandas as pd
    # 为了避免 akshare 特殊的代理阻断，我们在运行时清理代理环境变量
    import os
    for _proxy_key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
        os.environ.pop(_proxy_key, None)
    
    run_real_ingestion("600519")
