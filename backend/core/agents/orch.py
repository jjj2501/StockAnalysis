import json
import logging
from backend.core.data import DataFetcher
from backend.core.llm import get_llm_client
from backend.core.agents.prompts import (
    MACRO_ANALYST_PROMPT, 
    QUANT_RESEARCHER_PROMPT, 
    RISK_CONTROL_PROMPT,
    PORTFOLIO_MANAGER_PROMPT
)
from backend.core.agents.base import ReActAgent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, provider="ollama", model_name="qwen3:1.7b"):
        self.fetcher = DataFetcher()
        self.llm = get_llm_client(provider=provider, model_name=model_name)
        
        # 实例化基于 ReAct 循环的大语言模型数字班底
        self.agents = [
            ReActAgent(name="Macro Analyst", role_prompt=MACRO_ANALYST_PROMPT, llm=self.llm, max_turns=3),
            ReActAgent(name="Quant Researcher", role_prompt=QUANT_RESEARCHER_PROMPT, llm=self.llm, max_turns=3),
            ReActAgent(name="Risk Control Agent", role_prompt=RISK_CONTROL_PROMPT, llm=self.llm, max_turns=3)
        ]
        
    def run_safari(self, symbol: str):
        """流式迭代器：支持全链路 SSE 和动态工具调用的 ReAct 议会模式"""
        def _sse_pack(role, status, content, is_chunk=False, raw_data=None):
            payload = {
                "role": role,
                "status": status,
                "content": content,
                "is_chunk": is_chunk
            }
            if raw_data:
                payload["raw_data"] = raw_data
            return f"data: {json.dumps(payload)}\n\n"

        # ==========================================
        # Phase 1: 数据工程师打底，下发底层原始 JSON
        # ==========================================
        yield _sse_pack("Data Engineer", "typing", f"正在下潜至数据深渊，构建全维市场雷达探出 {symbol} 的底牌...\n\n")
        
        try:
            factors = self.fetcher.get_comprehensive_factors(symbol)
        except Exception as e:
            logger.error(f"Data Engineer 抓取崩溃: {e}")
            yield _sse_pack("Data Engineer", "error", f"致命错误：获取 {symbol} 的因子矩阵失败 ({str(e)})。")
            return
            
        yield _sse_pack("Data Engineer", "done", f"成功提取 {symbol} 的早期盘面。交由风控委员会进行查证研讨。\n", raw_data=factors)
        
        # 定义系统的中央议事大厅记忆体
        context_messages = [
            {"role": "user", "content": f"投资目标: {symbol}。 以下是数据工程师初采的底层面板参考 (部分系统工具无法触及的基础资料):\n{json.dumps(factors, ensure_ascii=False)}"}
        ]
        
        # ==========================================
        # Phase 2: 分析师开始接力登台，调用 MCP 工具或反驳
        # ==========================================
        for agent in self.agents:
            yield _sse_pack(agent.name, "typing", f"\n")
            
            final_content = ""
            for step in agent.run(context_messages, symbol):
                step_type = step.get("type")
                step_content = step.get("content", "")
                
                if step_type == "thought":
                    yield _sse_pack(agent.name, "typing", step_content + "\n\n", is_chunk=True)
                    final_content += (step_content + "\n\n")
                elif step_type == "action":
                    tool = step.get("tool")
                    args_str = json.dumps(step.get("args", {}))
                    msg = f"> ⚠️ **发起了底层数据调阅**: `{tool}({args_str})`\n\n"
                    yield _sse_pack(agent.name, "typing", msg, is_chunk=True)
                    final_content += msg
                elif step_type == "observation":
                    msg = f"> 📡 **收到系统数据探针响应**: `{step_content}`\n\n"
                    yield _sse_pack(agent.name, "typing", msg, is_chunk=True)
                    final_content += msg
                elif step_type == "final_answer":
                    yield _sse_pack(agent.name, "typing", step_content + "\n", is_chunk=True)
                    final_content += (step_content + "\n")
                elif step_type == "error":
                    yield _sse_pack(agent.name, "error", step_content, is_chunk=True)
                elif step_type == "system_warning":
                    yield _sse_pack(agent.name, "typing", f"\n[系统插断]: *{step_content}*\n", is_chunk=True)
            
            yield _sse_pack(agent.name, "done", "")
            
            # 把当前专家的结论性发言或整个脑回路存入历史长廊，传递给下一位开路的专家！
            context_messages.append({"role": "assistant", "name": agent.name, "content": final_content})

        # ==========================================
        # Phase 3: The Boss (Portfolio Manager) 进行最后裁定
        # ==========================================
        yield _sse_pack("Portfolio Manager", "typing", "")
        
        boss_context = "回顾先前的唇枪舌战记录：\n"
        for msg in context_messages:
            if msg["role"] == "assistant":
                boss_context += f"\n[{msg.get('name', 'Analyst')}] 说：\n{msg['content']}\n---"
                
        pm_prompt = PORTFOLIO_MANAGER_PROMPT.format(
            base_data=f"资产代码: {symbol}",
            # 新版本去除了固定点对点的填充，要求他直接看 Context
            macro_opinion=boss_context, 
            quant_opinion="",
            risk_opinion=""
        )
        
        try:
            for chunk in self.llm.stream_generate(pm_prompt):
                yield _sse_pack("Portfolio Manager", "typing", chunk, is_chunk=True)
            yield _sse_pack("Portfolio Manager", "done", "\n*** 全体智能体议事归票完毕 ***\n")
        except Exception as e:
            yield _sse_pack("Portfolio Manager", "error", f"\n风控节点断裂: {e}\n\n", is_chunk=True)
