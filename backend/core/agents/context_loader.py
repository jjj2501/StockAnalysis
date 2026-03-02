"""
分级上下文加载器 (ContextLoader)

参照 OpenClaw 理念：
- 常驻上下文（每次必须加载）：角色人设 + 长期记忆 + 近两日短期记忆 + 全局规律
- 按需动态加载：角色专属技能（仅首次使用对应角色时读取）
- 目标：节约 LLM 上下文窗口，同时保持 Agent 的连续记忆能力
"""

import logging
from typing import List, Dict, Optional

from backend.core.agents import memory_fs
from backend.core.agents.rag_retriever import hybrid_search

logger = logging.getLogger(__name__)

# 每个 Agent 角色对应的技能文件名（在 .agents/skills/ 下）
_ROLE_SKILL_FILES: Dict[str, List[str]] = {
    "Macro Analyst":       ["common_errors"],   # 宏观分析师可访问公共技能
    "Quant Researcher":    ["common_errors"],
    "Risk Control Agent":  ["common_errors"],
    "Portfolio Manager":   ["common_errors"],
}


class ContextLoader:
    """
    分级上下文构建器。
    为每个 Agent 的每次推演按需组装最小化的 System Prompt 上下文。

    常驻（4 类）：
      1. 角色人设（system.md 等价）
      2. 股票长期记忆（memory.md）
      3. 近两日短期记忆（今天 + 昨天 .md）
      4. 全局跨标的记忆

    按需（1 类）：
      5. 角色专属技能文档（仅由对应角色加载）
    """

    def __init__(self, agent_name: str, role_prompt: str):
        self.agent_name = agent_name
        self.role_prompt = role_prompt
        # 缓存本次会话已加载的技能内容，避免重复 IO
        self._skills_cache: Optional[str] = None

    def build_system_context(self, symbol: str) -> str:
        """
        构建完整的 System 层上下文字符串，注入 ReActAgent 的 messages[system]。

        Args:
            symbol: 当前分析的股票代码

        Returns:
            完整 system prompt 字符串
        """
        parts = [self.role_prompt]

        # ── 常驻：长期记忆 ──
        long_term = memory_fs.read_long_term(symbol)
        if long_term:
            parts.append(
                f"\n\n---\n## 📚 {symbol.upper()} 长期记忆库\n"
                f"（以下是过往推演中提炼的关键规律，请严肃参考）\n\n{long_term}"
            )

        # ── 常驻：近两日短期记忆 ──
        recent = memory_fs.read_recent_daily(symbol, days=2)
        if recent:
            parts.append(
                f"\n\n---\n## 🗂️ {symbol.upper()} 近期推演记忆\n"
                f"（以下是昨天/今天的推演摘要，可帮助你延续判断）\n\n{recent}"
            )

        # ── 常驻：全局跨标的记忆 ──
        global_mem = memory_fs.read_global_memory()
        if global_mem:
            parts.append(
                f"\n\n---\n## 🌐 全局市场规律库\n"
                f"（以下是跨标的积累的通用量化规律）\n\n{global_mem}"
            )

        # ── 按需：角色专属技能文档 ──
        skills_text = self._load_skills_once()
        if skills_text:
            parts.append(
                f"\n\n---\n## 🛠️ 专属技能参考手册\n{skills_text}"
            )

        return "\n".join(parts)

    def _load_skills_once(self) -> str:
        """
        懒加载角色技能文档（每个 ContextLoader 实例只加载一次）。
        只读取白名单目录内的 SKILL.md，不允许外部注入。
        """
        if self._skills_cache is not None:
            return self._skills_cache

        skill_files = _ROLE_SKILL_FILES.get(self.agent_name, [])
        parts = []
        for skill_name in skill_files:
            content = memory_fs.read_skill_file(skill_name)
            if content:
                parts.append(content)
            else:
                logger.debug(f"[ContextLoader] 技能文件不存在或格式不合规: {skill_name}")

        self._skills_cache = "\n\n---\n".join(parts) if parts else ""
        return self._skills_cache


def build_memory_injection_message(symbol: str, query: str = None) -> Dict[str, str]:
    """
    构建一条作为 user message 注入历史上下文的记忆提示。
    结合常驻上下文（近期 .md + memory.md）和混合 RAG 检索结果。

    Args:
        symbol: 股票代码
        query: 检索查询词（默认用股票代码作为查询），用于 hybrid_search

    Returns:
        {"role": "user", "content": "...记忆摘要文本..."}
    """
    long_term = memory_fs.read_long_term(symbol)
    recent = memory_fs.read_recent_daily(symbol, days=2)
    global_mem = memory_fs.read_global_memory()

    parts = []
    if long_term:
        parts.append(f"📚 **{symbol.upper()} 长期规律**\n{long_term[:800]}")
    if recent:
        parts.append(f"🗂️ **近期推演摘要**\n{recent[:600]}")
    if global_mem:
        parts.append(f"🌐 **全局规律**\n{global_mem[:400]}")

    # ── 混合 RAG 检索：额外召回历史中与当前标的最相关的片段 ──
    # 使用 symbol 或调用方传入的自定义 query 作为检索意图
    rag_query = query or f"{symbol} 投资分析 量化因子 风险规律"
    try:
        rag_hits = hybrid_search(rag_query, symbol=symbol, top_k=3)
        if rag_hits:
            rag_text = "\n\n".join(f"- {hit[:300]}" for hit in rag_hits)
            parts.append(f"🔍 **混合 RAG 关联记忆（70% 语义 + 30% 关键词）**\n{rag_text}")
    except Exception as e:
        logger.debug(f"[ContextLoader] RAG 检索失败（降级为仅常驻上下文）: {e}")

    if not parts:
        return {}

    content = (
        f"[系统记忆注入 — {symbol.upper()}]\n\n"
        + "\n\n".join(parts)
        + "\n\n请参考以上历史记忆，开始本轮推演。"
    )
    return {"role": "user", "content": content}
