"""
多轮辩论共识引擎 (DebateEngine)
维护各 Agent 发言的公共黑板，根据关键词检测「分歧信号」，
并对外暴露「触发追加辩论」与「共识评分」接口。
"""

import json
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 触发追加辩论的关键词（任意一方出现即认为存在尚待厘清的重大分歧）
_CONFLICT_KEYWORDS = [
    "一票否决", "不同意", "泡沫", "严重高估", "强烈反对",
    "数据不支持", "值得商榷", "隐患重重", "熔断风险",
    "截然相反", "截然不同", "极度危险", "存疑"
]

# 乐观结论关键词（多空阵营判断用）
_BULLISH_KEYWORDS = ["买入", "做多", "上涨", "逢低建仓", "乐观", "低估", "顺风"]
_BEARISH_KEYWORDS = ["卖出", "做空", "下跌", "清仓", "悲观", "高估", "逆风", "一票否决"]


class SharedBlackboard:
    """
    所有 Agent 之间共享的公共黑板。
    按发言顺序记录每一轮发言，同时维护「多空计分板」。
    """
    def __init__(self):
        self.sessions: List[Dict] = []   # 发言历史列表
        self.debate_round: int = 0       # 当前辩论回合编号
        self.bullish_votes: int = 0
        self.bearish_votes: int = 0

    def post(self, agent_name: str, content: str, round_num: int = 0):
        """Agent 发言后调用，内容上黑板"""
        entry = {
            "agent": agent_name,
            "round": round_num,
            "content": content
        }
        self.sessions.append(entry)
        
        # 更新多空计分板
        content_lower = content.lower()
        bull = sum(1 for w in _BULLISH_KEYWORDS if w in content)
        bear = sum(1 for w in _BEARISH_KEYWORDS if w in content)
        self.bullish_votes += bull
        self.bearish_votes += bear
        logger.debug(f"[黑板] {agent_name} 发言，多方信号+{bull}，空方信号+{bear}")

    def detect_conflict(self) -> bool:
        """检测最近的发言中是否出现「分歧关键词」"""
        recent = self.sessions[-3:] if len(self.sessions) >= 3 else self.sessions
        for entry in recent:
            for kw in _CONFLICT_KEYWORDS:
                if kw in entry["content"]:
                    logger.info(f"[辩论引擎] 检测到分歧关键词「{kw}」，建议追加辩论回合")
                    return True
        return False

    def get_summary_for_agent(self, exclude_agent: Optional[str] = None) -> str:
        """
        生成供下一位发言 Agent 阅读的黑板摘要。
        可排除当前 Agent 自己之前的某条发言（避免自我引用过度）。
        """
        lines = []
        for entry in self.sessions:
            if exclude_agent and entry["agent"] == exclude_agent:
                continue
            lines.append(f"【{entry['agent']}·回合{entry['round']}】\n{entry['content']}")
        return "\n---\n".join(lines)

    def get_consensus_score(self) -> Dict:
        """返回当前多空共识得分（0~100，50 为中性）"""
        total = self.bullish_votes + self.bearish_votes
        if total == 0:
            score = 50
        else:
            score = round(self.bullish_votes / total * 100)
        
        if score >= 65:
            verdict = "多方主导"
        elif score <= 35:
            verdict = "空方主导"
        else:
            verdict = "多空均衡"
        
        return {
            "score": score,
            "verdict": verdict,
            "bullish_votes": self.bullish_votes,
            "bearish_votes": self.bearish_votes
        }

    def get_narrative_history(self) -> List[Dict]:
        """返回完整的叙事历史（供存入知识记忆）"""
        return [dict(entry) for entry in self.sessions]


class DebateEngine:
    """
    多轮辩论状态机。
    在单轮瀑布结束后，检查是否存在分歧，若有则运行追加回合。
    """
    def __init__(self, max_debate_rounds: int = 2):
        self.max_debate_rounds = max_debate_rounds
        self.board = SharedBlackboard()
        self.current_round = 0

    def record_speech(self, agent_name: str, content: str):
        """记录一次发言"""
        self.board.post(agent_name, content, self.current_round)

    def should_trigger_debate(self) -> bool:
        """判断是否需要触发追加辩论"""
        if self.current_round >= self.max_debate_rounds:
            return False
        return self.board.detect_conflict()

    def start_next_round(self) -> int:
        """进入下一轮辩论，返回当前回合编号"""
        self.current_round += 1
        return self.current_round

    def get_board_summary(self) -> str:
        """获取黑板内容摘要，注入下一回合的 context"""
        return self.board.get_summary_for_agent()

    def get_final_consensus(self) -> Dict:
        """获取最终共识报告"""
        return {
            "rounds_completed": self.current_round,
            "consensus": self.board.get_consensus_score(),
            "history": self.board.get_narrative_history()
        }
