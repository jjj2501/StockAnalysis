import pytest
import chromadb
from pathlib import Path
from backend.core.rag.vector_db import VectorDBManager
from backend.core.rag.ingestion import KnowledgeIngestion

@pytest.fixture
def temp_vdb(tmp_path):
    """构建一个完全隔离的持久化测试版 VectorDB"""
    import backend.core.rag.vector_db
    from chromadb import Documents, EmbeddingFunction, Embeddings

    original_path = backend.core.rag.vector_db.DB_PATH
    backend.core.rag.vector_db.DB_PATH = tmp_path
    
    # 正常初始化持久化类，让 chromadb 自己管好内部 http/sqlite 线程池
    manager = VectorDBManager(collection_name="test_reports")
    
    # 我们只拦截它的建表，强塞一个假模型，防止去外网下分词器
    class DummyEmbedding(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return [[float(len(doc)), float(sum(ord(c) for c in doc)), 1.0] for doc in input]
            
    # 删除原版 collection，用假模型重盖一座
    try:
        manager.client.delete_collection("test_reports")
    except Exception:
        pass
        
    manager.collection = manager.client.create_collection(
        name="test_reports",
        embedding_function=DummyEmbedding()
    )
    
    yield manager
    
    # 清理还原
    backend.core.rag.vector_db.DB_PATH = original_path

@pytest.fixture
def mock_ingestion(temp_vdb):
    """构建自带测试级 DB 的测试切割管线"""
    # 劫持全局 get_vector_db 的返回，指向 temp_vdb
    import backend.core.rag.ingestion
    original_get = backend.core.rag.ingestion.get_vector_db
    backend.core.rag.ingestion.get_vector_db = lambda: temp_vdb
    
    # 构建小颗粒度切割器用于测试
    pipeline = KnowledgeIngestion(chunk_size=100, chunk_overlap=20)
    yield pipeline
    
    backend.core.rag.ingestion.get_vector_db = original_get

def test_chunking_and_upsert(mock_ingestion, temp_vdb):
    test_text = """
    We are proud to announce our Q1 results. Our revenue grew by 20% to $10B.
    However, due to the unexpected delay in TSMC's 3nm chip yield, our gross margins dropped to 40%.
    We expect this to be resolved in the latter half of the year.
    """
    
    # 模拟导入 TSMC 相关研报
    success = mock_ingestion.process_and_store_document("AAPL", "earnings", test_text)
    assert success is True, "Document ingestion should return True on success."
    
    # 测试检索
    docs = temp_vdb.query_similar("AAPL", "Why did gross margin drop?", n_results=1)
    
    assert len(docs) > 0, "Should retrieve at least 1 document chunk."
    # 检索出的切片应当命中 TSMC 或者 dropped 等关键词
    assert "margin" in docs[0].lower() or "tsmc" in docs[0].lower(), "Retrieved chunk should contain reason for margin drop."

def test_query_filtering(temp_vdb):
    """测试多资产环境下 metadata 隔离能力"""
    # 直接向 DB 插入双重数据
    temp_vdb.upsert_chunks("MSFT", "report", ["MSFT Azure cloud grew 30%"])
    temp_vdb.upsert_chunks("GOOG", "report", ["GOOG Cloud grew 28%"])
    
    # 我们查问云计算增长，但强制指定公司是 GOOG
    docs = temp_vdb.query_similar("GOOG", "cloud growth rate", n_results=1)
    
    assert len(docs) == 1
    assert "GOOG" in docs[0], "The query must strictly filter by the provided symbol metadata."
