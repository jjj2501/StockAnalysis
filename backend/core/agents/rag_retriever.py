"""
混合 RAG 检索引擎 (HybridRAGRetriever)

参照 OpenClaw 架构：70% 向量语义 + 30% BM25 关键词字面量加权融合。

- 向量检索（70%）：使用 ChromaDB + SentenceTransformer 理解语义
- BM25 检索（30%）：精确匹配数字、代码、错误名、股票代码等字面量
- 融合策略：RRF（Reciprocal Rank Fusion）+ 自定义权重

数据源：从 backend/data/agent_memory/ 目录下所有 Markdown 文件构建索引。
"""

import os
import re
import logging
import hashlib
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── 全局缓存 BM25 索引，避免每次推演都重建 ──
_bm25_index = None
_bm25_corpus: List[Dict] = []  # [{"id": str, "text": str, "symbol": str, "date": str}]
_bm25_built_at: Optional[datetime] = None
_BM25_TTL_MINUTES = 30  # 索引 TTL：30 分钟过期后重建

# 记忆根目录（与 memory_fs.py 保持一致）
_MEMORY_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "agent_memory"
)


def _tokenize(text: str) -> List[str]:
    """
    简单中英文混合分词器。
    - 英文：按空格/标点分词
    - 中文：单字分词（BM25 对中文逐字切效果尚可）
    """
    # 统一小写，切掉 Markdown 符号
    text = re.sub(r"[#*`>\-|]", " ", text).lower()
    # 英文词元
    tokens = re.findall(r"[a-z0-9][a-z0-9._%-]*", text)
    # 中文字符逐字拆分
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return tokens + chinese_chars


def _parse_markdown_to_chunks(file_path: str, symbol: str, date: str) -> List[Dict]:
    """
    将一个 Markdown 记忆文件切分为段落级别的 chunk。
    按 '---' 分隔符或二级标题（## ）分段。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    # 按 '---' 分隔符分段
    blocks = re.split(r"\n---+\n", content)
    chunks = []
    for i, block in enumerate(blocks):
        stripped = block.strip()
        if len(stripped) < 20:  # 过短块跳过
            continue
        chunk_id = f"{symbol}_{date}_{i}_{hashlib.md5(stripped.encode()).hexdigest()[:8]}"
        chunks.append({
            "id": chunk_id,
            "text": stripped,
            "symbol": symbol,
            "date": date,
            "source": file_path
        })
    return chunks


def _rebuild_bm25_index():
    """
    扫描所有 Markdown 记忆文件，重建 BM25 索引。
    会缓存在模块级全局变量中。
    """
    global _bm25_index, _bm25_corpus, _bm25_built_at

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("[RAG] rank_bm25 未安装，BM25 检索不可用。请运行: .\\uv pip install rank-bm25")
        return

    if not os.path.exists(_MEMORY_ROOT):
        return

    corpus = []
    # 遍历所有 {SYMBOL}/ 子目录的 .md 文件
    for entry in os.scandir(_MEMORY_ROOT):
        if not entry.is_dir():
            continue
        symbol = entry.name
        for md_file in os.scandir(entry.path):
            if not md_file.name.endswith(".md"):
                continue
            date = md_file.name.replace(".md", "")
            chunks = _parse_markdown_to_chunks(md_file.path, symbol, date)
            corpus.extend(chunks)

    if not corpus:
        logger.debug("[RAG] 记忆文件为空，BM25 索引未构建")
        return

    # 构建词元化语料
    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in corpus]
    _bm25_corpus = corpus
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_built_at = datetime.now()
    logger.info(f"[RAG] BM25 索引重建完成，共 {len(corpus)} 个 chunk")


def _get_bm25_index():
    """懒加载 + TTL 过期自动重建"""
    global _bm25_built_at
    if _bm25_index is None or (
        _bm25_built_at and
        (datetime.now() - _bm25_built_at).seconds > _BM25_TTL_MINUTES * 60
    ):
        _rebuild_bm25_index()
    return _bm25_index, _bm25_corpus


def bm25_search(query: str, symbol: str = None, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    BM25 关键词检索。
    返回: [(chunk_text, score), ...]，按得分降序。

    Args:
        query: 查询文本
        symbol: 可选，限定股票代码过滤
        top_k: 返回结果数量
    """
    index, corpus = _get_bm25_index()
    if index is None or not corpus:
        return []

    query_tokens = _tokenize(query)
    scores = index.get_scores(query_tokens)

    # 构建 (index, score) 列表并排序
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in indexed_scores:
        if score <= 0:
            break
        chunk = corpus[idx]
        # 可选：按股票过滤
        if symbol and chunk["symbol"].upper() != symbol.upper() and chunk["symbol"] != "global":
            continue
        results.append((chunk["text"], float(score)))
        if len(results) >= top_k:
            break

    return results


def vector_search(query: str, symbol: str = None, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    向量语义检索（ChromaDB + SentenceTransformer）。
    复用现有 VectorDBManager，返回: [(chunk_text, score), ...]
    """
    try:
        from backend.core.rag.vector_db import get_vector_db
        db = get_vector_db()
        if symbol:
            docs = db.query_similar(symbol, query, n_results=top_k)
        else:
            # 不限标的时查询所有（通过不传 where 过滤）
            results = db.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            docs = results.get("documents", [[]])[0]

        # ChromaDB 距离越小越好，转换为相似度
        return [(doc, 1.0 - i * 0.1) for i, doc in enumerate(docs)]

    except Exception as e:
        logger.debug(f"[RAG] 向量检索失败（可能是向量库为空）: {e}")
        return []


def _rrf_fuse(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    RRF（Reciprocal Rank Fusion）加权融合。
    k: RRF 平滑参数（默认 60，防止高排名结果权重过高）

    分数公式:
        score = vector_weight * (1/(k+rank_v)) + bm25_weight * (1/(k+rank_b))
    """
    scores: Dict[str, float] = {}

    # 向量结果：排名权重
    for rank, (text, _) in enumerate(vector_results):
        key = text[:200]  # 用文本前 200 字作为摘要 key
        scores[key] = scores.get(key, 0) + vector_weight / (k + rank + 1)
        scores[f"_text_{key}"] = text  # 保存完整文本

    # BM25 结果：排名权重
    for rank, (text, _) in enumerate(bm25_results):
        key = text[:200]
        scores[key] = scores.get(key, 0) + bm25_weight / (k + rank + 1)
        scores[f"_text_{key}"] = text

    # 按融合分数排序，剔除 _text_ 键
    fused = [
        (scores[f"_text_{k}"], v)
        for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not k.startswith("_text_") and f"_text_{k}" in scores
    ]
    return fused


def hybrid_search(
    query: str,
    symbol: str = None,
    top_k: int = 5,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[str]:
    """
    主入口：混合 RAG 检索。

    合并向量（语义）和 BM25（精确）检索结果，
    按 70/30 权重 RRF 融合，返回 top_k 条最相关文本片段。

    Args:
        query: 检索查询，通常是当前推演的标的+主要问题
        symbol: 限定股票代码（None 表示全局检索）
        top_k: 最终返回片段数量
        vector_weight: 向量搜索权重（默认 0.7）
        bm25_weight: BM25 权重（默认 0.3）

    Returns:
        List[str]：最相关的 Markdown 文本片段列表
    """
    fetch_k = min(top_k * 3, 15)  # 各路检索候选数量

    # ── 并行检索（同步） ──
    bm25_results = bm25_search(query, symbol=symbol, top_k=fetch_k)
    vector_results = vector_search(query, symbol=symbol, top_k=fetch_k)

    # ── RRF 加权融合 ──
    if not bm25_results and not vector_results:
        return []

    if not vector_results:
        # 纯 BM25 降级
        return [text for text, _ in bm25_results[:top_k]]
    if not bm25_results:
        # 纯向量降级
        return [text for text, _ in vector_results[:top_k]]

    fused = _rrf_fuse(vector_results, bm25_results, vector_weight, bm25_weight)
    return [text for text, _ in fused[:top_k]]


def refresh_index():
    """强制重建 BM25 索引（推演写入新记忆后可调用）"""
    global _bm25_built_at
    _bm25_built_at = None  # 让 TTL 检查触发重建
    _get_bm25_index()
