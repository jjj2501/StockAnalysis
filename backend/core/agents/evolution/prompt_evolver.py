"""
Prompt 自进化引擎 (PromptEvolver)

基于推演绩效反馈，自动优化 prompts.py 中各角色的 Prompt。

安全约束:
1. 只在 Prompt 末尾的 `# [自进化经验区]` 标记后追加
2. 不删除/修改原始 Prompt 内容
3. 每次最多追加 1 条经验
4. Prompt 总经验条目数上限 10 条（超出淘汰最旧的）
"""

import os
import re
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# prompts.py 的绝对路径
_PROMPTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "prompts.py"
)

# 进化标记（在此标记之后追加经验条目）
_EVOLUTION_MARKER = "# [自进化经验区]"

# 每个角色最多保留的经验条目数
_MAX_INSIGHTS_PER_ROLE = 10

# 角色变量名映射
ROLE_TO_VAR = {
    "Macro Analyst": "MACRO_ANALYST_PROMPT",
    "Quant Researcher": "QUANT_RESEARCHER_PROMPT",
    "Risk Control Agent": "RISK_CONTROL_PROMPT",
    "Portfolio Manager": "PORTFOLIO_MANAGER_PROMPT",
}


def generate_evolution_insight(
    llm,
    agent_name: str,
    symbol: str,
    debate_summary: str,
    verdict: str,
    consensus_score: int
) -> Optional[str]:
    """
    调用 LLM 生成一条经验进化条目。

    Args:
        llm: LLM 客户端实例
        agent_name: Agent 角色名
        symbol: 本次推演的股票代码
        debate_summary: 辩论摘要
        verdict: 最终裁定
        consensus_score: 共识评分 (0-100)

    Returns:
        经验条目字符串（≤60字），失败返回 None
    """
    prompt = (
        f"你是一位 AI 系统的自我优化引擎。刚才 {agent_name} 参与了关于 {symbol} 的投研推演。\n"
        f"共识评分: {consensus_score}/100\n"
        f"辩论要点: {debate_summary[:500]}\n"
        f"最终裁定: {verdict[:300]}\n\n"
        f"请基于本次推演过程，为 {agent_name} 总结出一条简短的经验教训或注意事项，"
        f"帮助其在未来推演中表现更好。\n\n"
        f"要求:\n"
        f"- 必须以 '⚡' 开头\n"
        f"- 不超过 60 个字\n"
        f"- 必须是具体可执行的建议，不要泛泛而谈\n"
        f"- 示例: '⚡ 当RSI>70且北向资金连续3日流出时，必须降低看多权重'\n\n"
        f"只输出经验条目本身:"
    )

    try:
        result = ""
        for chunk in llm.stream_generate(prompt):
            result += chunk
        result = result.strip()

        # 安全校验：空结果
        if not result or len(result) < 5:
            logger.warning(f"[PromptEvolver] LLM 返回结果过短或为空: '{result}'")
            return None

        # 安全校验：不能太长，不能包含代码
        if len(result) > 100:
            result = result[:100]

        # 安全校验：拒绝包含代码注入关键字
        if any(kw in result for kw in ["import ", "def ", "class ", "exec(", "eval("]):
            logger.warning(f"[PromptEvolver][安全] 检测到代码注入，拒绝: {result[:50]}")
            return None

        # 安全校验：拒绝包含网络异常/错误堆栈的虚假经验（LLM 连接失败时会产生）
        error_markers = [
            "HTTPConnectionPool", "ConnectionError", "Traceback",
            "Max retries exceeded", "TimeoutError", "refused",
            "智能体核心错误", "Error:", "Exception"
        ]
        if any(marker in result for marker in error_markers):
            logger.warning(f"[PromptEvolver][安全] 检测到异常文本混入经验，拒绝: {result[:80]}")
            return None

        if not result.startswith("⚡"):
            result = "⚡ " + result

        return result
    except Exception as e:
        logger.error(f"[PromptEvolver] 生成经验失败: {e}")
        return None


def read_current_insights(agent_name: str) -> List[str]:
    """
    读取指定角色当前已有的自进化经验条目。
    """
    var_name = ROLE_TO_VAR.get(agent_name)
    if not var_name:
        return []

    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    # 从对应变量的文本中提取 ⚡ 开头的行
    # 找到变量区间
    pattern = rf'{var_name}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []

    prompt_text = match.group(1)
    insights = [line.strip() for line in prompt_text.split("\n") if line.strip().startswith("⚡")]
    return insights


def inject_insight_to_prompt(agent_name: str, new_insight: str) -> bool:
    """
    将新经验条目安全注入到 prompts.py 中对应角色的 Prompt 内。

    安全机制:
    - 只在 `# [自进化经验区]` 标记之后追加
    - 超出上限时淘汰最旧条目
    - 不修改标记之前的原始 Prompt 内容

    Returns:
        是否成功
    """
    var_name = ROLE_TO_VAR.get(agent_name)
    if not var_name:
        logger.error(f"[PromptEvolver] 未知角色: {agent_name}")
        return False

    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"[PromptEvolver] 读取 prompts.py 失败: {e}")
        return False

    # 找到对应变量的 Prompt 文本块
    pattern = rf'({var_name}\s*=\s*""")(.*?)(""")'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        logger.error(f"[PromptEvolver] 未找到 {var_name} 定义")
        return False

    prefix = match.group(1)
    prompt_body = match.group(2)
    suffix = match.group(3)

    # 检查是否存在进化标记
    if _EVOLUTION_MARKER not in prompt_body:
        logger.error(f"[PromptEvolver] {var_name} 中缺少进化标记 '{_EVOLUTION_MARKER}'")
        return False

    # 分割：标记前（原始 Prompt）和标记后（经验区）
    parts = prompt_body.split(_EVOLUTION_MARKER, 1)
    original_part = parts[0]
    evolution_part = parts[1] if len(parts) > 1 else ""

    # 提取已有经验条目
    existing_insights = [
        line.strip() for line in evolution_part.split("\n")
        if line.strip().startswith("⚡")
    ]

    # 检查是否重复
    if new_insight.strip() in existing_insights:
        logger.info(f"[PromptEvolver] 经验已存在，跳过: {new_insight[:40]}")
        return False

    # 添加新经验
    existing_insights.append(new_insight.strip())

    # 超出上限时淘汰最旧的
    if len(existing_insights) > _MAX_INSIGHTS_PER_ROLE:
        removed = existing_insights.pop(0)
        logger.info(f"[PromptEvolver] 淘汰最旧经验: {removed[:40]}")

    # 重建经验区文本
    insights_text = "\n".join(f"- {ins}" for ins in existing_insights)
    new_evolution_part = f"\n{insights_text}\n"

    # 重建完整 Prompt
    new_prompt_body = f"{original_part}{_EVOLUTION_MARKER}{new_evolution_part}"
    new_content = content[:match.start()] + prefix + new_prompt_body + suffix + content[match.end():]

    # 安全写入
    try:
        with open(_PROMPTS_FILE, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info(f"[PromptEvolver] ✅ 已为 {agent_name} 注入新经验: {new_insight[:50]}")
        return True
    except Exception as e:
        logger.error(f"[PromptEvolver] 写入 prompts.py 失败: {e}")
        return False


def evolve_prompt(
    llm,
    agent_name: str,
    symbol: str,
    debate_summary: str,
    verdict: str,
    consensus_score: int
) -> Dict:
    """
    完整的 Prompt 进化流程:
    1. LLM 生成经验条目
    2. 注入到 prompts.py
    3. 通过 Git 提交到远程

    Returns:
        {"success": bool, "insight": str, "git_result": dict}
    """
    result = {"success": False, "insight": "", "git_result": {}}

    # 1. 生成经验
    insight = generate_evolution_insight(
        llm, agent_name, symbol, debate_summary, verdict, consensus_score
    )
    if not insight:
        result["insight"] = "（LLM 未能生成有效经验）"
        return result

    result["insight"] = insight

    # 2. 注入 Prompt
    if not inject_insight_to_prompt(agent_name, insight):
        return result

    # 3. Git 提交
    try:
        from backend.core.agents.evolution.git_ops import full_evolution_cycle
        git_result = full_evolution_cycle(
            file_path="backend/core/agents/prompts.py",
            agent_name=agent_name,
            insight=insight
        )
        result["git_result"] = git_result
    except Exception as e:
        logger.warning(f"[PromptEvolver] Git 操作失败（经验已写入但未提交）: {e}")
        result["git_result"] = {"success": False, "message": str(e)}

    result["success"] = True
    return result
