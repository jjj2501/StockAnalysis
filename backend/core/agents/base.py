import json
import logging
from typing import List, Dict, Any, Generator
from backend.core.llm import OllamaLLM
from backend.core.agents.tools import mcp_tools, AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

class ReActAgent:
    """
    基于 Tool Calling (MCP) 的动态自治智能体基类。
    不再是单一的 Prompt-Completion 映射，而是一个完整的 ReAct (Reason + Act) 循环机器。
    """
    def __init__(self, name: str, role_prompt: str, llm: OllamaLLM, max_turns: int = 5):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm
        self.max_turns = max_turns
        # 内置系统级可用抓手
        self.tools = mcp_tools
        self.tool_functions = AVAILABLE_TOOLS

    def run(self, context_messages: List[Dict[str, str]], symbol: str) -> Generator[Dict[str, Any], None, None]:
        """
        启动智能体的自我推导与查证引擎。
        通过 yield 向外广播其每一步的脑内活动（Thought）、调用的工具（Action）以及最终结论。
        
        Args:
            context_messages: 历史长廊上下文（包含了其他 Agent 的辩论与发言记录）
            symbol: 标的代码，用于在必要时让 Agent 传递给 Tool
        """
        # 构建当前 Agent 私有的对话内存，注入其人设
        messages = [{"role": "system", "content": self.role_prompt}]
        messages.extend(context_messages)
        
        # 强制提醒当前研讨的核心标的
        messages.append({
            "role": "user", 
            "content": f"请针对标的 {symbol} 发表你的专业研判。如果有同行之前的发言，请务必仔细审视并毫不客气地进行赞同、反驳或补充。如果你认为现有数据不足以让你下定论，请调用系统提供的 MCP 数据库探针工具获取更多指标。"
        })

        turn_count = 0
        while turn_count < self.max_turns:
            turn_count += 1
            
            # 阻塞式调用 LLM 获取本轮的响应
            response_msg = self.llm.chat_with_tools(messages=messages, tools=self.tools)
            
            if not response_msg:
                yield {"type": "error", "content": f"{self.name} 神经中枢停止响应。"}
                break
                
            messages.append(response_msg)
            
            # 1. 检查 Agent 在这一回合是否说出了任何“推导思考过程”
            if response_msg.get("content"):
                yield {"type": "thought", "content": response_msg["content"]}
                
            # 2. 检查 Agent 是否请求了 Tool Calling动作
            tool_calls = response_msg.get("tool_calls", [])
            
            if not tool_calls:
                # 如果既没有调用工具，且有文本输出，说明它已经得出了最终结论，可以跳出循环结束回合
                yield {"type": "final_answer", "content": response_msg.get("content", "")}
                break

            # 3. 如果请求了Action，则执行函数提取Observation
            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                tool_call_id = tool_call.get("id")
                
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"])
                except Exception:
                    fn_args = {}
                
                # 强力防呆补丁：某些小尺寸模型在 JSON Schema 约束下偶尔还是会漏传必选参数
                # 我们这里直接将调度层面的 symbol 兜底注入
                if "symbol" not in fn_args or not fn_args["symbol"]:
                    fn_args["symbol"] = symbol

                yield {"type": "action", "tool": fn_name, "args": fn_args}
                
                # 在真实沙盘里执行该 MCP 工具
                if fn_name in TOOL_REGISTRY:
                    try:
                        observation = TOOL_REGISTRY[fn_name](**fn_args)
                    except Exception as e:
                        observation = f"探针执行崩溃: {str(e)}"
                else:
                    observation = f"系统中尚未组装名为 '{fn_name}' 的 MCP 模组。"
                
                # 记录执行结果，并作为 "tool" role 送回给大模型进行下回合推断
                yield {"type": "observation", "content": observation[:100] + "...(数据量大已截断)" if len(observation) > 100 else observation}
                
                tool_msg = {
                    "role": "tool",
                    "content": observation
                }
                if tool_call_id:
                    tool_msg["tool_call_id"] = tool_call_id
                
                messages.append(tool_msg)
        
        if turn_count >= self.max_turns:
            yield {"type": "system_warning", "content": f"{self.name} 陷入了太深的逻辑死循环，被主持人强行拉回现实。"}
