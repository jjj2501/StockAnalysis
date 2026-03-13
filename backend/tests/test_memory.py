"""
智能体知识记忆系统（AgentMemoryStore）单元测试
使用 pytest tmp_path 临时目录，不污染实际 data/ 目录。
"""

import json
import os
import pytest

from backend.core.agents import memory as mem_module
from backend.core.agents.memory import AgentMemoryStore


@pytest.fixture(autouse=True)
def patch_memory_dir(tmp_path, monkeypatch):
    """
    将 _MEMORY_DIR 重定向到临时目录，保证测试彼此隔离，
    同时不会向真实 backend/data/agent_memory/ 写数据。
    """
    tmp_dir = str(tmp_path / "agent_memory")
    monkeypatch.setattr(mem_module, "_MEMORY_DIR", tmp_dir)
    yield tmp_dir


class TestSaveAndLoadSession:
    """会话保存与读取测试"""

    def test_save_and_load_single_session(self):
        """保存一次推演后，load_history 应能读取到结果"""
        AgentMemoryStore.save_session(
            symbol="AAPL",
            session_id="test_001",
            debate_history=[{"agent": "Macro Analyst", "round": 0, "content": "测试发言"}],
            final_verdict="建议持有，中性评级",
            consensus={"score": 55, "verdict": "多空均衡", "bullish_votes": 2, "bearish_votes": 1}
        )

        history = AgentMemoryStore.load_history("AAPL", top_k=5)
        assert len(history) == 1
        assert history[0]["consensus"]["score"] == 55
        assert "建议持有" in history[0]["verdict_summary"]

    def test_load_history_empty_when_no_file(self):
        """没有记录时，load_history 应返回空列表"""
        result = AgentMemoryStore.load_history("NONEXISTENT", top_k=5)
        assert result == []

    def test_save_multiple_sessions_and_top_k(self):
        """保存多次推演，top_k 参数应正确截断返回条目"""
        for i in range(5):
            AgentMemoryStore.save_session(
                symbol="600519",
                session_id=f"sess_{i}",
                debate_history=[],
                final_verdict=f"第 {i} 次裁决",
                consensus={"score": 50 + i, "verdict": "多空均衡", "bullish_votes": i, "bearish_votes": i}
            )

        history = AgentMemoryStore.load_history("600519", top_k=3)
        assert len(history) == 3
        # 应该是最近 3 条（即第 2、3、4 次推演）
        assert "第 4 次裁决" in history[-1]["verdict_summary"]

    def test_auto_trim_to_20_sessions(self):
        """超过 20 次推演后，文件应只保留最近 20 条"""
        for i in range(25):
            AgentMemoryStore.save_session(
                symbol="TRIMTEST",
                session_id=f"s_{i}",
                debate_history=[],
                final_verdict=f"裁决 {i}",
                consensus={"score": 50, "verdict": "多空均衡", "bullish_votes": 1, "bearish_votes": 1}
            )

        history = AgentMemoryStore.load_history("TRIMTEST", top_k=30)
        assert len(history) <= 20

    def test_symbol_uppercase_normalization(self):
        """symbol 不区分大小写，保存后应都归一到大写文件名"""
        AgentMemoryStore.save_session(
            symbol="aapl",
            session_id="lower_001",
            debate_history=[],
            final_verdict="小写代码测试",
            consensus={"score": 60, "verdict": "多方主导", "bullish_votes": 3, "bearish_votes": 1}
        )

        history = AgentMemoryStore.load_history("AAPL", top_k=5)
        assert len(history) == 1
        assert "小写代码测试" in history[0]["verdict_summary"]


class TestSaveAndLoadInsights:
    """全局智慧库存写与读取测试"""

    def test_save_and_get_insights(self):
        """保存一条洞见后，get_global_insights 应能读取到"""
        AgentMemoryStore.save_insight("高 RSI + 外资净流出通常预示短期回调", tags=["technical", "northbound"])

        insights = AgentMemoryStore.get_global_insights(top_k=10)
        assert len(insights) == 1
        assert "高 RSI" in insights[0]["content"]
        assert "technical" in insights[0]["tags"]

    def test_insights_empty_when_no_file(self):
        """没有洞见文件时，应返回空列表"""
        result = AgentMemoryStore.get_global_insights(top_k=10)
        assert result == []

    def test_multiple_insights_top_k(self):
        """保存多条洞见后，top_k 参数应正确截断"""
        for i in range(5):
            AgentMemoryStore.save_insight(f"规律 {i}", tags=[f"tag_{i}"])

        insights = AgentMemoryStore.get_global_insights(top_k=3)
        assert len(insights) == 3
        # 应取最后 3 条
        assert "规律 4" in insights[-1]["content"]

    def test_insight_auto_trim_to_200(self):
        """超过 200 条洞见时，应只保留最近 200 条"""
        for i in range(210):
            AgentMemoryStore.save_insight(f"洞见 {i}")

        insights = AgentMemoryStore.get_global_insights(top_k=300)
        assert len(insights) <= 200


class TestBuildMemoryContext:
    """build_memory_context 综合输出测试"""

    def test_returns_empty_string_when_no_history(self):
        """无历史记忆时，应返回空字符串"""
        context = AgentMemoryStore.build_memory_context("UNKNOWN")
        assert context == ""

    def test_returns_context_with_history(self):
        """有历史记录时，context 应包含历史推演信息"""
        AgentMemoryStore.save_session(
            symbol="TEST",
            session_id="ctx_001",
            debate_history=[],
            final_verdict="维持中性，等待突破",
            consensus={"score": 50, "verdict": "多空均衡", "bullish_votes": 1, "bearish_votes": 1}
        )

        context = AgentMemoryStore.build_memory_context("TEST")
        assert "TEST" in context
        assert "历史推演记忆" in context

    def test_context_includes_global_insights(self):
        """有全局洞见时，context 也应包含智慧库内容"""
        AgentMemoryStore.save_insight("降息叠加外资流入往往是中长线买入信号")
        AgentMemoryStore.save_session(
            symbol="WITHINS",
            session_id="ins_001",
            debate_history=[],
            final_verdict="结论：谨慎看多",
            consensus={"score": 60, "verdict": "多方主导", "bullish_votes": 3, "bearish_votes": 1}
        )

        context = AgentMemoryStore.build_memory_context("WITHINS")
        assert "全局智慧库" in context
        assert "降息叠加外资流入" in context


class TestListAllSymbols:
    """list_all_symbols 索引测试"""

    def test_empty_when_no_data(self):
        """没有任何会话文件时，应返回空列表"""
        result = AgentMemoryStore.list_all_symbols()
        assert result == []

    def test_lists_saved_symbols(self):
        """保存多只股票后，应能列出全部代码"""
        for sym in ["AAPL", "TSLA", "600519"]:
            AgentMemoryStore.save_session(
                symbol=sym,
                session_id=f"{sym}_001",
                debate_history=[],
                final_verdict="测试",
                consensus={"score": 50, "verdict": "多空均衡", "bullish_votes": 1, "bearish_votes": 1}
            )

        symbols = AgentMemoryStore.list_all_symbols()
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        assert "600519" in symbols
        assert len(symbols) == 3
