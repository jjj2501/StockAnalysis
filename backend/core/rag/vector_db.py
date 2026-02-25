import chromadb
from chromadb.config import Settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 数据存储在 backend/data/vector_db 下
DB_PATH = Path(__file__).parent.parent.parent / "data" / "vector_db"

class VectorDBManager:
    """管理 ChromaDB 实例与财报向量库的核心类"""
    
    def __init__(self, collection_name: str = "financial_reports"):
        DB_PATH.mkdir(parents=True, exist_ok=True)
        # 初始化持久化 Chroma 客户端
        # 必须显式关闭所有外联遥测（Telemetry）及更新检查，防止代理握手 EOF 错误
        self.client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            )
        )
        # 用默认存在的检索模型建表
        try:
            from chromadb.utils import embedding_functions
            
            # 优先检查本地是否己下载模型，实现纯脱机加载
            local_model_path = DB_PATH.parent / "models" / "all-MiniLM-L6-v2"
            model_name_or_path = str(local_model_path) if local_model_path.exists() else "all-MiniLM-L6-v2"
            
            # 使用检测到的路径或名称实力化 Embedding Function
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name_or_path)
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
                metadata={"description": "Financial reports and transcripts"}
            )
            logger.info(f"Vector DB collection '{collection_name}' loaded. DB Path: {DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to load Vector DB collection: {e}")
            raise
            
    def upsert_chunks(self, symbol: str, source_type: str, chunks: list[str], chunk_ids: list[str] = None):
        """将切片后的知识库写入指定标的的存储区"""
        if not chunks:
            return
            
        if chunk_ids is None:
            import hashlib
            # 生成确定性ID以覆盖旧数据
            chunk_ids = [f"{symbol}_{source_type}_{hashlib.md5(c.encode()).hexdigest()[:8]}" for c in chunks]
            
        metadatas = [{"symbol": symbol.upper(), "type": source_type} for _ in chunks]
        
        self.collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        logger.info(f"Upserted {len(chunks)} chunks for {symbol} ({source_type})")

    def query_similar(self, symbol: str, query_text: str, n_results: int = 3) -> list:
        """根据查询意图，查找该标的相关的财报切片"""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"symbol": symbol.upper()} # 精确过滤只能是这个股票的黑匣子
            )
            
            # results['documents'] 是一个列表的列表
            if results and 'documents' in results and results['documents'][0]:
                return results['documents'][0]
            return []
        except Exception as e:
            logger.error(f"Error querying vector db for {symbol}: {e}")
            return []

# 全局单例
_db_manager = None
def get_vector_db() -> VectorDBManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = VectorDBManager()
    return _db_manager
