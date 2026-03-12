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


class TestCrossComments:
    """交叉评论机制测试套件（v3.0 新增）"""

    def test_post_cross_comment(self):
        """交叉评论能正确写入黑板并计入多空票数"""
        board = SharedBlackboard()
        board.post_cross_comment(
            "Macro Analyst", "Risk Control Agent",
            "你的风控结论过于悲观，买入信号明显"
        )

        assert len(board.cross_comments) == 1
        assert board.cross_comments[0]["from_agent"] == "Macro Analyst"
        assert board.cross_comments[0]["to_agent"] == "Risk Control Agent"
        assert board.bullish_votes > 0  # "买入" 关键词

    def test_get_comments_on_agent(self):
        """能正确获取针对特定 Agent 的评论"""
        board = SharedBlackboard()
        board.post_cross_comment("A", "B", "你的分析有盲点")
        board.post_cross_comment("C", "B", "我同意 A 的质疑")
        board.post_cross_comment("A", "C", "你的数据来源存疑")

        comments_on_b = board.get_comments_on("B")
        assert len(comments_on_b) == 2

        comments_on_c = board.get_comments_on("C")
        assert len(comments_on_c) == 1

    def test_unreplied_comments(self):
        """未回复评论过滤正确"""
        board = SharedBlackboard()
        board.post_cross_comment("A", "B", "你的分析有问题")
        board.post_cross_comment("C", "B", "同意 A 的看法")

        # 初始都未回复
        unreplied = board.get_unreplied_comments_on("B")
        assert len(unreplied) == 2

        # 标记一条已回复后
        board.mark_comment_responded("A", "B")
        unreplied = board.get_unreplied_comments_on("B")
        assert len(unreplied) == 1
        assert unreplied[0]["from_agent"] == "C"

    def test_cross_review_context(self):
        """能正确为 Agent 组装交叉评论上下文"""
        board = SharedBlackboard()
        board.post("A", "A 的发言", round_num=0)
        board.post("B", "B 的发言", round_num=0)
        board.post_cross_comment("B", "A", "A 你的论据不足")

        context = board.get_cross_review_context("A")
        assert "B 的发言" in context
        assert "A 你的论据不足" in context
        assert "A 的发言" not in context  # 不该包含自己的发言

    def test_should_continue_dialogue(self):
        """未回复评论含分歧关键词时应继续对话"""
        board = SharedBlackboard()
        board.post_cross_comment("A", "B", "你的结论数据不支持，存疑")

        # 有未回复的含分歧关键词的评论
        assert board.should_continue_dialogue() is True

        # 标记回复后不再触发
        board.mark_comment_responded("A", "B")
        assert board.should_continue_dialogue() is False

    def test_should_not_continue_without_conflict(self):
        """未回复评论不含分歧关键词时不应继续对话"""
        board = SharedBlackboard()
        board.post_cross_comment("A", "B", "我赞同你的中性分析")
        assert board.should_continue_dialogue() is False

    def test_get_peers_speeches(self):
        """能正确获取除自身以外的同行发言"""
        board = SharedBlackboard()
        board.post("A", "A 的发言", round_num=0)
        board.post("B", "B 的发言", round_num=0)
        board.post("C", "C 的发言", round_num=0)

        peers = board.get_peers_speeches("B")
        assert "A" in peers
        assert "C" in peers
        assert "B" not in peers

    def test_narrative_history_includes_cross_comments(self):
        """叙事历史应包含交叉评论"""
        board = SharedBlackboard()
        board.post("A", "A 的发言", round_num=0)
        board.post_cross_comment("B", "A", "不同意 A 的观点")

        history = board.get_narrative_history()
        assert len(history) == 2
        assert history[1]["type"] == "cross_comment"


class TestDebateEngineCrossReview:
    """DebateEngine 交叉评论轮次管理测试套件（v3.0 新增）"""

    def test_should_trigger_cross_review(self):
        """有 >= 2 名 Agent 发言时应触发交叉评论"""
        engine = DebateEngine(max_cross_review_rounds=1)
        engine.record_speech("A", "A 的分析")
        engine.record_speech("B", "B 的分析")

        assert engine.should_trigger_cross_review() is True

    def test_should_not_trigger_cross_review_single_agent(self):
        """只有 1 名 Agent 发言时不应触发"""
        engine = DebateEngine(max_cross_review_rounds=1)
        engine.record_speech("A", "A 的分析")

        assert engine.should_trigger_cross_review() is False

    def test_should_not_exceed_max_cross_review_rounds(self):
        """超过最大轮次后不应继续触发"""
        engine = DebateEngine(max_cross_review_rounds=1)
        engine.record_speech("A", "A 的分析")
        engine.record_speech("B", "B 的分析")
        engine.start_cross_review_round()

        assert engine.should_trigger_cross_review() is False

    def test_record_cross_comment(self):
        """DebateEngine 记录交叉评论正常"""
        engine = DebateEngine()
        engine.record_cross_comment("A", "B", "不赞同 B 的结论")

        assert len(engine.board.cross_comments) == 1

    def test_final_consensus_includes_cross_review_rounds(self):
        """最终共识报告应包含交叉评论轮次数"""
        engine = DebateEngine(max_cross_review_rounds=2)
        engine.record_speech("A", "买入信号明显")
        engine.start_cross_review_round()

        result = engine.get_final_consensus()
        assert "cross_review_rounds" in result
        assert result["cross_review_rounds"] == 1
