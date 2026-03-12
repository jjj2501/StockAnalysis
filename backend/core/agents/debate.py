"""
多轮辩论共识引擎 (DebateEngine) v3.0
维护各 Agent 发言的公共黑板，根据关键词检测「分歧信号」，
并对外暴露「触发追加辩论」与「共识评分」接口。

v3.0 新增：
  - 交叉评论机制：Agent 可定向点评其他 Agent 的观点
  - 多轮对话管理：支持 Agent 之间的回合制互评对话
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
    按发言顺序记录每一轮发言，同时维护「多空计分板」和「交叉评论区」。
    """
    def __init__(self):
        self.sessions: List[Dict] = []         # 发言历史列表
        self.cross_comments: List[Dict] = []   # 交叉评论列表
        self.debate_round: int = 0             # 当前辩论回合编号
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

    # ── 交叉评论系统 ──

    def post_cross_comment(self, from_agent: str, to_agent: str,
                           comment: str, round_num: int = 0):
        """
        Agent 发表对其他 Agent 的定向交叉评论。

        Args:
            from_agent: 发表评论的 Agent
            to_agent: 被评论的 Agent
            comment: 评论内容
            round_num: 交叉评论所在的轮次号
        """
        entry = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "comment": comment,
            "round": round_num,
            "responded": False  # 标记被评论方是否已回应
        }
        self.cross_comments.append(entry)

        # 交叉评论也计入多空计分板
        bull = sum(1 for w in _BULLISH_KEYWORDS if w in comment)
        bear = sum(1 for w in _BEARISH_KEYWORDS if w in comment)
        self.bullish_votes += bull
        self.bearish_votes += bear
        logger.debug(
            f"[黑板·交叉评论] {from_agent} → {to_agent}，"
            f"多方信号+{bull}，空方信号+{bear}"
        )

    def get_comments_on(self, agent_name: str) -> List[Dict]:
        """获取所有针对指定 Agent 的交叉评论"""
        return [c for c in self.cross_comments if c["to_agent"] == agent_name]

    def get_unreplied_comments_on(self, agent_name: str) -> List[Dict]:
        """获取所有未回复的、针对指定 Agent 的交叉评论"""
        return [
            c for c in self.cross_comments
            if c["to_agent"] == agent_name and not c["responded"]
        ]

    def mark_comment_responded(self, from_agent: str, to_agent: str):
        """标记来自 from_agent 对 to_agent 的评论为已回应"""
        for c in self.cross_comments:
            if c["from_agent"] == from_agent and c["to_agent"] == to_agent:
                c["responded"] = True

    def get_cross_review_context(self, agent_name: str) -> str:
        """
        为特定 Agent 组装交叉评论上下文摘要。
        包含：所有同行的发言 + 已有的针对该 Agent 的评论。
        """
        lines = []

        # 1. 其他同行的原始发言
        lines.append("## 同行发言记录")
        for entry in self.sessions:
            if entry["agent"] != agent_name:
                lines.append(
                    f"【{entry['agent']}·回合{entry['round']}】\n{entry['content']}"
                )

        # 2. 针对该 Agent 的交叉评论
        comments_on_me = self.get_comments_on(agent_name)
        if comments_on_me:
            lines.append("\n## 针对你的同行评论")
            for c in comments_on_me:
                lines.append(
                    f"【来自 {c['from_agent']}·回合{c['round']}】\n{c['comment']}"
                )

        return "\n---\n".join(lines)

    def get_peers_speeches(self, agent_name: str) -> Dict[str, str]:
        """
        获取除指定 Agent 以外的所有 Agent 的最新发言，
        返回 {agent_name: speech_content} 字典。
        """
        peers = {}
        for entry in self.sessions:
            if entry["agent"] != agent_name:
                # 同名 Agent 取最新发言
                peers[entry["agent"]] = entry["content"]
        return peers

    def should_continue_dialogue(self) -> bool:
        """
        判断是否需要继续多轮对话。
        条件：存在未回复的评论 且 评论中含有分歧信号。
        """
        unreplied = [c for c in self.cross_comments if not c["responded"]]
        if not unreplied:
            return False

        # 检查未回复评论中是否含有分歧关键词
        for c in unreplied:
            for kw in _CONFLICT_KEYWORDS:
                if kw in c["comment"]:
                    logger.info(
                        f"[辩论引擎] 交叉评论中检测到分歧关键词「{kw}」"
                        f"({c['from_agent']} → {c['to_agent']})，建议继续对话"
                    )
                    return True
        return False

    def get_cross_comment_summary(self) -> str:
        """获取所有交叉评论的叙事摘要（供 PM 审阅）"""
        if not self.cross_comments:
            return ""
        lines = []
        for c in self.cross_comments:
            replied_tag = "✅已回应" if c["responded"] else "⏳待回应"
            lines.append(
                f"【{c['from_agent']} → {c['to_agent']}·"
                f"回合{c['round']}·{replied_tag}】\n{c['comment']}"
            )
        return "\n---\n".join(lines)

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
        包含原始发言 + 交叉评论摘要。
        """
        lines = []
        for entry in self.sessions:
            if exclude_agent and entry["agent"] == exclude_agent:
                continue
            lines.append(f"【{entry['agent']}·回合{entry['round']}】\n{entry['content']}")

        # 追加交叉评论摘要
        cross_summary = self.get_cross_comment_summary()
        if cross_summary:
            lines.append("\n## 交叉评论记录\n" + cross_summary)

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
        """返回完整的叙事历史（供存入知识记忆），包含交叉评论"""
        history = [dict(entry) for entry in self.sessions]
        # 将交叉评论也附加到历史中
        for c in self.cross_comments:
            history.append({
                "agent": f"{c['from_agent']} → {c['to_agent']}",
                "round": c["round"],
                "content": f"[交叉评论] {c['comment']}",
                "type": "cross_comment"
            })
        return history


class DebateEngine:
    """
    多轮辩论状态机 v3.0。
    在单轮瀑布结束后，检查是否存在分歧，若有则运行追加回合。
    新增：交叉评论轮次管理。
    """
    def __init__(self, max_debate_rounds: int = 2,
                 max_cross_review_rounds: int = 1):
        self.max_debate_rounds = max_debate_rounds
        self.max_cross_review_rounds = max_cross_review_rounds
        self.board = SharedBlackboard()
        self.current_round = 0
        self.cross_review_round = 0  # 当前交叉评论轮次

    def record_speech(self, agent_name: str, content: str):
        """记录一次发言"""
        self.board.post(agent_name, content, self.current_round)

    def record_cross_comment(self, from_agent: str, to_agent: str,
                             comment: str):
        """记录一条交叉评论"""
        self.board.post_cross_comment(
            from_agent, to_agent, comment, self.cross_review_round
        )

    def mark_responded(self, from_agent: str, to_agent: str):
        """标记交叉评论已被回应"""
        self.board.mark_comment_responded(from_agent, to_agent)

    def should_trigger_debate(self) -> bool:
        """判断是否需要触发追加辩论"""
        if self.current_round >= self.max_debate_rounds:
            return False
        return self.board.detect_conflict()

    def should_trigger_cross_review(self) -> bool:
        """判断是否需要触发交叉评论轮次"""
        if self.cross_review_round >= self.max_cross_review_rounds:
            return False
        # 只要有 >= 2 名 Agent 发言过就可以开启交叉评论
        unique_agents = set(e["agent"] for e in self.board.sessions)
        return len(unique_agents) >= 2

    def should_continue_dialogue(self) -> bool:
        """判断是否需要继续多轮对话（基于未回复评论）"""
        return self.board.should_continue_dialogue()

    def start_next_round(self) -> int:
        """进入下一轮辩论，返回当前回合编号"""
        self.current_round += 1
        return self.current_round

    def start_cross_review_round(self) -> int:
        """进入下一轮交叉评论，返回当前交叉评论轮次号"""
        self.cross_review_round += 1
        return self.cross_review_round

    def get_peers_speeches(self, agent_name: str) -> Dict[str, str]:
        """获取除指定 Agent 以外的所有同行的最新发言"""
        return self.board.get_peers_speeches(agent_name)

    def get_unreplied_comments_on(self, agent_name: str) -> List[Dict]:
        """获取指定 Agent 的未回复评论"""
        return self.board.get_unreplied_comments_on(agent_name)

    def get_cross_review_context(self, agent_name: str) -> str:
        """获取为特定 Agent 组装的交叉评论上下文"""
        return self.board.get_cross_review_context(agent_name)

    def get_board_summary(self) -> str:
        """获取黑板内容摘要，注入下一回合的 context"""
        return self.board.get_summary_for_agent()

    def get_final_consensus(self) -> Dict:
        """获取最终共识报告"""
        return {
            "rounds_completed": self.current_round,
            "cross_review_rounds": self.cross_review_round,
            "consensus": self.board.get_consensus_score(),
            "history": self.board.get_narrative_history()
        }
