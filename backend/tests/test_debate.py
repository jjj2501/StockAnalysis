"""
辩论引擎（DebateEngine / SharedBlackboard）单元测试
"""

import pytest
from backend.core.agents.debate import SharedBlackboard, DebateEngine


class TestSharedBlackboard:
    """SharedBlackboard 公共黑板测试套件"""

    def test_post_and_history(self):
        """发言上黑板后，能正确检索到历史"""
        board = SharedBlackboard()
        board.post("Macro Analyst", "宏观数据显示通胀压力较大", round_num=0)
        board.post("Quant Researcher", "RSI 超买，技术面偏空", round_num=0)

        history = board.get_narrative_history()
        assert len(history) == 2
        assert history[0]["agent"] == "Macro Analyst"
        assert history[1]["agent"] == "Quant Researcher"

    def test_bullish_bearish_votes(self):
        """多空关键词计分正常工作"""
        board = SharedBlackboard()
        board.post("Macro Analyst", "建议买入，逢低建仓机会明显", round_num=0)
        board.post("Quant Researcher", "虽然技术面偏弱，但长线看好", round_num=0)
        board.post("Risk Control Agent", "卖出信号强烈，熔断风险显著，一票否决", round_num=0)

        # 多方票数应大于 0（买入 + 逢低建仓 + 看好）
        assert board.bullish_votes > 0
        # 空方票数应大于 0（卖出 + 一票否决）
        assert board.bearish_votes > 0

    def test_consensus_score_bullish(self):
        """全多方发言时，评分应偏向多方"""
        board = SharedBlackboard()
        for _ in range(3):
            board.post("A", "买入 逢低建仓 看好 乐观", round_num=0)
        result = board.get_consensus_score()
        assert result["score"] >= 50

    def test_consensus_score_bearish(self):
        """全空方发言时，评分应偏向空方"""
        board = SharedBlackboard()
        for _ in range(3):
            board.post("A", "卖出 清仓 高估 悲观 做空", round_num=0)
        result = board.get_consensus_score()
        assert result["score"] <= 50

    def test_detect_conflict_with_keywords(self):
        """包含分歧关键词时，应检测到冲突"""
        board = SharedBlackboard()
        board.post("Macro Analyst", "基本面良好，乐观展望", round_num=0)
        board.post("Risk Control Agent", "存疑！数据不支持现有估值", round_num=0)

        assert board.detect_conflict() is True

    def test_detect_conflict_without_keywords(self):
        """不含分歧关键词时，不应触发冲突检测"""
        board = SharedBlackboard()
        board.post("Macro Analyst", "利率保持平稳，短期无大变化", round_num=0)
        board.post("Quant Researcher", "RSI 在合理区间内", round_num=0)

        assert board.detect_conflict() is False

    def test_get_summary_excludes_agent(self):
        """get_summary_for_agent 能正确排除指定 Agent 的发言"""
        board = SharedBlackboard()
        board.post("A", "A 的发言", round_num=0)
        board.post("B", "B 的发言", round_num=0)
        board.post("C", "C 的发言", round_num=0)

        summary = board.get_summary_for_agent(exclude_agent="B")
        assert "B 的发言" not in summary
        assert "A 的发言" in summary
        assert "C 的发言" in summary

    def test_consensus_no_votes_returns_50(self):
        """没有任何多空关键词时，评分应为 50（中性）"""
        board = SharedBlackboard()
        board.post("A", "这是一段没有倾向性的中性描述。", round_num=0)
        result = board.get_consensus_score()
        assert result["score"] == 50
        assert result["verdict"] == "多空均衡"


class TestDebateEngine:
    """DebateEngine 多轮辩论状态机测试套件"""

    def test_initial_state(self):
        """初始状态：回合数为 0，黑板为空"""
        engine = DebateEngine(max_debate_rounds=2)
        assert engine.current_round == 0
        assert len(engine.board.sessions) == 0

    def test_record_speech(self):
        """记录发言后，黑板上应有对应条目"""
        engine = DebateEngine()
        engine.record_speech("Macro Analyst", "宏观分析：降息概率上升")
        engine.record_speech("Quant Researcher", "量化因子指向超买区间")

        history = engine.board.get_narrative_history()
        assert len(history) == 2

    def test_should_not_trigger_when_max_round_reached(self):
        """到达最大辩论回合数时，should_trigger_debate 应返回 False"""
        engine = DebateEngine(max_debate_rounds=1)
        engine.current_round = 1  # 模拟已达最大回合
        engine.record_speech("Risk Control Agent", "一票否决，存疑，熔断风险")

        # 即使有冲突关键词，也不应触发（已到限）
        assert engine.should_trigger_debate() is False

    def test_should_trigger_with_conflict_keywords(self):
        """有分歧关键词且未到最大回合时，应触发追加辩论"""
        engine = DebateEngine(max_debate_rounds=2)
        engine.record_speech("Risk Control Agent", "截然相反的结论，数据不支持当前估值")

        assert engine.should_trigger_debate() is True

    def test_start_next_round(self):
        """start_next_round 应递增回合编号并返回新回合号"""
        engine = DebateEngine(max_debate_rounds=2)
        new_round = engine.start_next_round()
        assert new_round == 1
        assert engine.current_round == 1

        new_round2 = engine.start_next_round()
        assert new_round2 == 2
        assert engine.current_round == 2

    def test_get_final_consensus_structure(self):
        """get_final_consensus 应返回包含 rounds_completed、consensus、history 的字典"""
        engine = DebateEngine(max_debate_rounds=2)
        engine.record_speech("Macro Analyst", "买入")
        engine.start_next_round()
        engine.record_speech("Quant Researcher", "卖出")

        result = engine.get_final_consensus()
        assert "rounds_completed" in result
        assert "consensus" in result
        assert "history" in result
        assert result["rounds_completed"] == 1
        assert isinstance(result["history"], list)
        assert len(result["history"]) == 2

    def test_board_summary_contains_all_speeches(self):
        """get_board_summary 应包含所有已记录的发言内容"""
        engine = DebateEngine()
        engine.record_speech("Macro Analyst", "宏观面数据支持看多")
        engine.record_speech("Risk Control Agent", "波动率锥形态异常")

        summary = engine.get_board_summary()
        assert "宏观面数据支持看多" in summary
        assert "波动率锥形态异常" in summary
