import json
from unittest.mock import patch, MagicMock
from backend.core.agents.orch import AgentOrchestrator

@patch('backend.core.agents.base.ReActAgent.respond_to_comments')
@patch('backend.core.agents.base.ReActAgent.review_peers')
@patch('backend.core.agents.base.ReActAgent.run')
@patch('backend.core.agents.orch.DataFetcher')
@patch('backend.core.agents.orch.get_llm_client')
@patch('backend.core.agents.orch.read_long_term', return_value="")
@patch('backend.core.agents.orch.read_recent_daily', return_value="")
@patch('backend.core.agents.orch.read_global_memory', return_value="")
@patch('backend.core.agents.orch.build_memory_injection_message', return_value=[])
@patch('backend.core.agents.orch.write_session_to_daily')
@patch('backend.core.agents.orch.promote_to_long_term')
def test_orch_yields_new_ui_events(
    mock_promote, mock_write, mock_build, mock_gl, mock_rd, mock_rl, 
    mock_llm, mock_fetcher, mock_run_method, mock_review_peers_method, mock_respond_method
):
    """
    测试 run_safari 能正确发送 phase_update 和 rebuttal (带 comment_quotes) 事件
    """
    # 模拟数据采集
    mock_fetcher.return_value.get_comprehensive_factors.return_value = {"mock_factor": 1.0}
    
    # 模拟 LLM 客户端
    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance
    
    # 模拟 PM 的 JSON 响应
    mock_response = MagicMock()
    mock_response.message.content = '{"score": 80, "verdict": "买入", "reasoning": "不错"}'
    mock_llm_instance.chat.return_value = mock_response

    # 配置 ReActAgent 的类级别生成器 Mock
    def mock_run_impl(*args, **kwargs):
        yield {"type": "done", "content": "mock speech"}
        
    def mock_review_peers_impl(*args, **kwargs):
        yield {"type": "cross_comment_done", "content": "mock comment"}
        
    def mock_respond_to_comments_impl(*args, **kwargs):
        yield {"type": "rebuttal_done", "content": "mock rebuttal"}

    mock_run_method.side_effect = mock_run_impl
    mock_review_peers_method.side_effect = mock_review_peers_impl
    mock_respond_method.side_effect = mock_respond_to_comments_impl

    orch = AgentOrchestrator(max_debate_rounds=0)
    
    # 运行流式推演
    events = []
    for chunk in orch.run_safari("MOCK_SYMBOL"):
        if chunk.startswith("data: "):
            try:
                payload = json.loads(chunk[len("data: "):])
                events.append(payload)
            except json.JSONDecodeError:
                pass

    # 1. 测试 phase_update 阶段事件覆盖率
    phase_events = [e for e in events if e.get("event") == "phase_update"]
    phases_yielded = [e.get("phase") for e in phase_events]
    
    # 验证是否推送了 1-6 全阶段
    assert 1 in phases_yielded
    assert 2 in phases_yielded
    assert 3 in phases_yielded
    assert 4 in phases_yielded
    assert 5 in phases_yielded
    assert 6 in phases_yielded
    
    # 2. 测试 rebuttal 事件是否带回 comment_quotes 引用
    rebuttal_events = [e for e in events if e.get("event") == "rebuttal"]
    
    # 至少应该有一个反驳事件（只要有交叉评论产生）
    assert len(rebuttal_events) > 0
    
    for r_event in rebuttal_events:
        assert "comment_quotes" in r_event
        # 因为模拟的交叉评论内容是 "mock comment"，预期它会携带在引用里
        quotes = r_event["comment_quotes"]
        assert isinstance(quotes, list)
        if len(quotes) > 0:
            assert "text" in quotes[0]
            assert "mock comment" in quotes[0]["text"]
