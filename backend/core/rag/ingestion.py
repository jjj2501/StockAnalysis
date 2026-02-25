import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.rag.vector_db import get_vector_db

logger = logging.getLogger(__name__)

class KnowledgeIngestion:
    """财报长文本自动切块并置入向量库流水线"""
    
    def __init__(self, chunk_size=800, chunk_overlap=150):
        # 建立切片器，采用典型的长篇非结构化文章重叠断点策略
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        self.vdb = get_vector_db()
        
    def process_and_store_document(self, symbol: str, source_type: str, raw_text: str):
        """核心处理方法：接收长文 -> 降噪切片 -> 丢给 VectorDB 打标建立索引"""
        if not raw_text or len(raw_text.strip()) < 10:
            logger.warning(f"Ignored ingestion for {symbol} due to empty/too short document.")
            return False
            
        try:
            # 执行智能切块
            chunks = self.text_splitter.split_text(raw_text)
            logger.info(f"Symbol {symbol} [{source_type}] document split into {len(chunks)} chunks.")
            
            # 使用 ChromaDB 原生封装方法推入存储表
            self.vdb.upsert_chunks(
                symbol=symbol,
                source_type=source_type,
                chunks=chunks
            )
            return True
            
        except Exception as e:
            logger.error(f"Ingestion failed for {symbol}: {e}")
            return False

# 测试用例 Mock入口
def ingest_mock_data():
    """灌入一条测试数据，模拟长篇公告或 Earnings Call"""
    mock_transcript_nvda = """
[NVDA 2026 Q1 Earnings Call Transcript]
CEO Jensen Huang: "The AI supercycle is in full swing. Our Blackwell architecture GPUs are seeing unprecedented demand from major hyperscalers. 
However, we acknowledge that supply chain bottlenecks, particularly in advanced packaging (CoWoS) from TSMC, remain our primary constraint for the next two quarters. 
We expect to clear this backlog by Q4, translating to an estimated $42B revenue in that quarter. Gross margins dipped slightly to 74.8% due to early ramp costs."
"""
    
    pipeline = KnowledgeIngestion(chunk_size=500, chunk_overlap=50)
    pipeline.process_and_store_document("NVDA", "earnings_call", mock_transcript_nvda)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_mock_data()
