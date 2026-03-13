"""
智能体知识记忆系统 (AgentMemoryStore)
持久化每次推演记录，支持按股票读取历史摘要，
并积累跨股票的通用规律作为全局智慧库。
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 默认存储路径
_MEMORY_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "agent_memory"
)


def _ensure_dir():
    os.makedirs(_MEMORY_DIR, exist_ok=True)


class AgentMemoryStore:
    """
    轻量级 JSON 文件持久化记忆系统。

    目录结构:
        backend/data/agent_memory/
            {SYMBOL}_sessions.json  — 每只股票的历史推演会话
            global_insights.json   — 跨股票提炼的通用规律库
    """

    # ─────────────────────────────────────────────
    # 写入接口
    # ─────────────────────────────────────────────

    @staticmethod
    def save_session(
        symbol: str,
        session_id: str,
        debate_history: List[Dict],
        final_verdict: str,
        consensus: Dict
    ):
        """
        推演结束后保存完整会话记录。

        Args:
            symbol: 股票代码（如 "AAPL"）
            session_id: 本次推演唯一 ID（时间戳字符串即可）
            debate_history: DebateEngine.get_final_consensus()["history"]
            final_verdict: Portfolio Manager 最终裁决的完整文本
            consensus: 多空共识计分字典
        """
        _ensure_dir()
        path = os.path.join(_MEMORY_DIR, f"{symbol.upper()}_sessions.json")

        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            entry = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "debate_history": debate_history,
                "verdict": final_verdict,
                "consensus": consensus
            }
            data.append(entry)

            # 只保留最近 20 次推演
            if len(data) > 20:
                data = data[-20:]

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"[记忆存储] 已保存 {symbol} 推演记录 session={session_id}")
        except Exception as e:
            logger.error(f"[记忆存储] 保存失败: {e}")

    @staticmethod
    def save_insight(insight: str, tags: List[str] = None):
        """
        向全局智慧库写入一条新规律洞见。

        Args:
            insight: 洞见文本（如"高RSI+外资净流出历史上多对应短期回调"）
            tags: 分类标签列表（如 ["technical", "northbound"]）
        """
        _ensure_dir()
        path = os.path.join(_MEMORY_DIR, "global_insights.json")

        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    library = json.load(f)
            else:
                library = []

            entry = {
                "id": f"ins_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "content": insight,
                "tags": tags or []
            }
            library.append(entry)

            # 保留最近 200 条
            if len(library) > 200:
                library = library[-200:]

            with open(path, "w", encoding="utf-8") as f:
                json.dump(library, f, ensure_ascii=False, indent=2)

            logger.info(f"[全局智慧库] 新增洞见: {insight[:60]}...")
        except Exception as e:
            logger.error(f"[全局智慧库] 写入失败: {e}")

    # ─────────────────────────────────────────────
    # 读取接口
    # ─────────────────────────────────────────────

    @staticmethod
    def load_history(symbol: str, top_k: int = 3) -> List[Dict]:
        """
        读取该股票最近 top_k 次推演的摘要（仅返回 verdict 与时间）。
        用于推演前注入历史上下文，让 Agent 有"历史记忆"。
        """
        path = os.path.join(_MEMORY_DIR, f"{symbol.upper()}_sessions.json")
        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 取最近 top_k 条，仅保留轻量字段
            recent = data[-top_k:] if len(data) >= top_k else data
            return [
                {
                    "timestamp": e.get("timestamp", ""),
                    "verdict_summary": e.get("verdict", "")[:300],
                    "consensus": e.get("consensus", {})
                }
                for e in recent
            ]
        except Exception as e:
            logger.error(f"[记忆读取] 读取 {symbol} 历史失败: {e}")
            return []

    @staticmethod
    def get_global_insights(top_k: int = 10) -> List[Dict]:
        """读取全局智慧库最近 top_k 条洞见"""
        path = os.path.join(_MEMORY_DIR, "global_insights.json")
        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                library = json.load(f)
            return library[-top_k:]
        except Exception as e:
            logger.error(f"[全局智慧库] 读取失败: {e}")
            return []

    @staticmethod
    def build_memory_context(symbol: str) -> str:
        """
        为推演注入历史记忆，构建可直接放入 context_messages 的字符串。
        包含: 1. 该股票历史结论摘要  2. 全局智慧库节选
        """
        history = AgentMemoryStore.load_history(symbol, top_k=3)
        insights = AgentMemoryStore.get_global_insights(top_k=5)

        parts = []

        if history:
            parts.append(f"📂 **{symbol} 历史推演记忆**（最近 {len(history)} 次）：")
            for i, h in enumerate(history, 1):
                ts = h.get("timestamp", "")[:10]
                consensus = h.get("consensus", {})
                verdict = h.get("verdict_summary", "无记录")
                score = consensus.get("score", 50)
                verdict_label = consensus.get("verdict", "")
                parts.append(
                    f"  [{i}] {ts} 多空共识={score}分({verdict_label})\n"
                    f"      结论摘要: {verdict[:150]}..."
                )

        if insights:
            parts.append("\n🧠 **全局智慧库节选**（经多次推演积累的通用规律）：")
            for ins in insights:
                parts.append(f"  • {ins['content']}")

        if not parts:
            return ""

        return "\n".join(parts)

    @staticmethod
    def list_all_symbols() -> List[str]:
        """列出所有有历史记忆的股票代码"""
        if not os.path.exists(_MEMORY_DIR):
            return []
        files = os.listdir(_MEMORY_DIR)
        symbols = [
            f.replace("_sessions.json", "")
            for f in files
            if f.endswith("_sessions.json")
        ]
        return sorted(symbols)
