"""
OpenClaw 式 Markdown 文件记忆管理器 (MemoryFS)

目录结构:
    backend/data/agent_memory/
    ├── {SYMBOL}/
    │   ├── memory.md          # 长期记忆（LLM 提炼的关键洞见，永久保留）
    │   ├── 2026-03-01.md      # 短期记忆（当日推演压缩摘要，自动追加）
    │   └── 2026-02-28.md
    └── global/
        ├── memory.md          # 跨标的全局长期记忆
        └── 2026-03-01.md      # 全局当日摘要
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# 记忆根目录
_MEMORY_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "agent_memory"
)

# 技能文件同名白名单根目录（只读）
_SKILLS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".agents", "skills")
)


def _symbol_dir(symbol: str) -> str:
    """获取股票专属记忆目录（确保存在）"""
    d = os.path.join(_MEMORY_ROOT, symbol.upper())
    os.makedirs(d, exist_ok=True)
    return d


def _global_dir() -> str:
    """获取全局记忆目录"""
    d = os.path.join(_MEMORY_ROOT, "global")
    os.makedirs(d, exist_ok=True)
    return d


# ──────────────────────────────────────────────
# 写入接口（短期记忆 / 长期记忆）
# ──────────────────────────────────────────────

def write_session_to_daily(symbol: str, markdown_summary: str, is_global: bool = False) -> None:
    """
    将本次推演摘要追加写入今日的短期记忆文件。
    文件名格式: YYYY-MM-DD.md
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if is_global:
        target_dir = _global_dir()
    else:
        target_dir = _symbol_dir(symbol)

    file_path = os.path.join(target_dir, f"{today}.md")
    timestamp = datetime.now().strftime("%H:%M")

    content = f"\n---\n\n## {today} {timestamp} | {symbol.upper()} 推演摘要\n\n{markdown_summary.strip()}\n"

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[MemoryFS] 短期记忆已追加: {file_path}")
    except Exception as e:
        logger.error(f"[MemoryFS] 写入短期记忆失败: {e}")


def promote_to_long_term(symbol: str, insight: str, is_global: bool = False) -> None:
    """
    将提炼出的关键洞见追加写入长期记忆文件 memory.md。
    长期记忆是永久性的，不会自动清理。
    """
    if is_global:
        target_dir = _global_dir()
    else:
        target_dir = _symbol_dir(symbol)

    file_path = os.path.join(target_dir, "memory.md")
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # 初始化文件头（首次创建时）
    if not os.path.exists(file_path):
        header = f"# {symbol.upper()} — 长期记忆库\n\n> 由 Agent 推演后自动提炼，仅记录高价值规律。\n\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header)

    content = f"\n- **[{timestamp}]** {insight.strip()}\n"

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[MemoryFS] 长期记忆已写入: {symbol} → {insight[:60]}")
    except Exception as e:
        logger.error(f"[MemoryFS] 写入长期记忆失败: {e}")


# ──────────────────────────────────────────────
# 读取接口（按需加载原则）
# ──────────────────────────────────────────────

def read_recent_daily(symbol: str, days: int = 2) -> str:
    """
    读取最近 N 天的短期记忆文件内容（今天 + 昨天）。
    这是常驻上下文的一部分，每次推演必须读取。
    """
    parts = []
    target_dir = _symbol_dir(symbol)

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        file_path = os.path.join(target_dir, f"{date}.md")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    parts.append(f"### 📅 {date} 短期记忆\n{content}")
            except Exception as e:
                logger.warning(f"[MemoryFS] 读取 {date} 短期记忆失败: {e}")

    return "\n\n".join(parts) if parts else ""


def read_long_term(symbol: str) -> str:
    """
    读取股票专属长期记忆（memory.md）。
    这是常驻上下文的一部分。
    """
    file_path = os.path.join(_symbol_dir(symbol), "memory.md")
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"[MemoryFS] 读取 {symbol} 长期记忆失败: {e}")
        return ""


def read_global_memory() -> str:
    """
    读取跨标的全局长期记忆（global/memory.md）及今日全局摘要。
    """
    parts = []
    g_dir = _global_dir()

    # 全局 memory.md
    mem_path = os.path.join(g_dir, "memory.md")
    if os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                parts.append(f"### 🧠 全局长期规律库\n{content}")
        except Exception as e:
            logger.warning(f"[MemoryFS] 读取全局记忆失败: {e}")

    # 全局今日摘要
    today = datetime.now().strftime("%Y-%m-%d")
    today_path = os.path.join(g_dir, f"{today}.md")
    if os.path.exists(today_path):
        try:
            with open(today_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                parts.append(f"### 📅 全局今日摘要\n{content}")
        except Exception as e:
            logger.warning(f"[MemoryFS] 读取全局今日摘要失败: {e}")

    return "\n\n".join(parts) if parts else ""


# ──────────────────────────────────────────────
# 技能文件安全读取（只读白名单）
# ──────────────────────────────────────────────

def read_skill_file(skill_name: str) -> Optional[str]:
    """
    【安全】从技能白名单目录读取技能文档（纯文本，只读）。
    - 只允许读取 .agents/skills/ 目录内的 .md 文件
    - 必须包含 YAML frontmatter name/description 字段
    - 禁止路径穿越（如 ../../etc/passwd）
    """
    # 路径安全校验：只允许白名单目录内文件
    safe_path = os.path.abspath(os.path.join(_SKILLS_ROOT, skill_name, "SKILL.md"))
    if not safe_path.startswith(_SKILLS_ROOT):
        logger.error(f"[MemoryFS][安全] 拒绝路径穿越攻击: {skill_name}")
        return None

    if not os.path.exists(safe_path):
        return None

    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()

        # YAML frontmatter 格式校验（至少包含 name 和 description）
        if "name:" not in content or "description:" not in content:
            logger.warning(f"[MemoryFS][安全] 技能文件格式不合规，已拒绝: {safe_path}")
            return None

        return content
    except Exception as e:
        logger.error(f"[MemoryFS] 读取技能文件失败: {e}")
        return None


def list_all_symbols() -> list:
    """列出所有有 Markdown 记忆的股票代码"""
    if not os.path.exists(_MEMORY_ROOT):
        return []
    return sorted([
        d for d in os.listdir(_MEMORY_ROOT)
        if os.path.isdir(os.path.join(_MEMORY_ROOT, d)) and d != "global"
    ])
