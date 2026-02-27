import abc
import requests
import json
import os
from typing import Optional

class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate_report(self, symbol: str, data: dict) -> str:
        pass

    def generate_backtest_report(self, symbol: str, strategy: str, metrics: dict) -> str:
        pass

    @abc.abstractmethod
    def generate_portfolio_risk_report(self, portfolio: list, risk_metrics: dict) -> str:
        pass

    @abc.abstractmethod
    def generate_factor_report_stream(self, symbol: str, factors_data: dict):
        """流式生成包含所有量化数据的深度剖析报告"""
        pass

    def _build_backtest_prompt(self, symbol: str, strategy: str, metrics: dict) -> str:
        return f"""
        你是一位资深的量化交易专家。请对股票 {symbol} 使用 {strategy} 策略的回测结果进行深度评价。
        
        【回测绩效指标】
        - 总收益率: {metrics.get('total_return_pct', 0):.2f}%
        - 年化收益率: {metrics.get('annual_return_pct', 0):.2f}%
        - 最大回撤: {metrics.get('max_drawdown_pct', 0):.2f}%
        - 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
        - 总交易次数: {metrics.get('total_trades', 0)}
        - 胜率: {metrics.get('win_rate', 0)*100:.2f}%
        
        【分析要求】
        1. **收益风险比评价**: 根据收益率和回撤、夏普比率，评价该策略在此历史时段的表现是否稳健。
        2. **策略适用性**: 简述该策略是否适合该股票的波动特性。
        3. **优化建议**: 针对回测表现，提出可能的优化方向（如调整参数、增加过滤条件等）。
        
        请用中文回答，保持专业客观。
        """

    def _build_portfolio_risk_prompt(self, portfolio: list, risk_metrics: dict) -> str:
        import json
        portfolio_str = ", ".join([f"{p['symbol']}({p.get('shares', 0)}股/市场:{p.get('market', 'CN')}/计价币种:{p.get('currency', 'CNY')})" for p in portfolio])
        
        metrics = risk_metrics.get("metrics", {})
        hist = metrics.get("historical", {})
        perf = risk_metrics.get("performance", {})
        corr = risk_metrics.get("correlation_matrix", {})
        
        corr_str = json.dumps(corr, ensure_ascii=False)
        
        return f"""
        你是一位极其严苛的机构级全球量化风控官。请根据以下个人投资组合的真实历史回测数据，用“无情且极其直白”的大白话，撰写一份专业的风控诊断及调仓药方报告。你的目标是打醒那些盲目自信的散户。
        
        【全球投资组合构成】
        - 包含资产与底牌地界: {portfolio_str}
        - 组合统合折算人民币总价值 (CNY): {risk_metrics.get('total_value', 0):.2f} 元
        - 组合年化综合波动率: {risk_metrics.get('annual_volatility', 0)*100:.2f}%
        
        【机构核心绩效与回撤指标 (所有海外资产均已被按日汇率强制穿透折算为统一的人民币净值)】
        - 过去一年真实最大回撤幅度: {perf.get('max_drawdown_pct', 0)*100:.2f}% 
        - 过去一年真实最大亏损金额: {perf.get('max_drawdown_amount', 0):.2f} RMB
        - 夏普比率 (Sharpe Ratio): {perf.get('sharpe_ratio', 0):.2f} (大于1优秀，负数说明不如存银行)
        
        【跨国别/跨币系相关性矩阵】(数值接近1代表不但同涨同跌，且连汇率曝险都一样)
        {corr_str}
        
        【分析要求】
        1. **极限拷问回撤**：指出绝对回撤金额，质问投资者在面临这种跨国/跨市场风暴时能否拿得住。
        2. **性价比与汇率风险批判**：不要被表面赚钱蒙蔽，由于所有资产已折算成人民币净值，请批判夏普比率。重点批评可能存在的“单边押注美元”等外汇汇率波动敞口过分集中的行为。
        3. **假分散的真面目**：利用中美大国博弈或不同周期的视角，根据皮尔逊相关性揪出看似买在海外，其实还是一损俱损的同质化标的。
        4. **【核心】调仓硬命令**: 直下药方。比如：“立刻大幅削减30%的集中度过高的人民币或美元资产，补充不同政治经济周期的市场标的或者加密数字防守资产。”

        请用 Markdown 格式，语气务必专业、犀利、像个恨铁不成钢的教官，绝不使用机械排比过渡词，字数控制在 500 字以内。
        """

    def _build_factor_prompt(self, symbol: str, data: dict) -> str:
        tech = data.get("technical", {})
        fund = data.get("fundamental", {})
        sent = data.get("sentiment", {})
        north = data.get("northbound", {})
        
        info_str = f"【财务基本面】\n市盈率PE: {fund.get('PE')}, 市净率PB: {fund.get('PB')}, 净资产收益率ROE: {fund.get('ROE')}%\n"
        info_str += f"市值规模: {fund.get('MarketCap')}\n\n"
        
        info_str += f"【短期技术面】\nRSI(相对强弱): {tech.get('RSI')}, 威廉指标(WR): {tech.get('WR')}, ROC(变动率): {tech.get('ROC')}\n"
        info_str += f"布林带位置: {tech.get('BB')}%\n\n"
        
        info_str += f"【市场情绪与资金】\n换手率: {sent.get('Turnover')}%\n"
        info_str += f"北向资金流动态: 当日买入 {north.get('NetBuy', '未知')}，外资总持股比 {north.get('HoldingRatio', '未知')}\n\n"
        
        info_str += f"【新闻舆情监测】\n"
        news_data = data.get("news", {})
        info_str += f"舆情情感分: {news_data.get('sentiment_score', '未知')} (满分100，越高越看多)\n"
        
        ai_pred = data.get("ai_prediction")
        if ai_pred:
            info_str += f"\n注: 其他深度学习AI子模型(LSTM等)对该股给出的独立倾向是: {ai_pred}\n"
        
        return f"""
        你是一位实战经验丰富且极其敏锐的多空对冲基金经理。请基于以下我摘录的关于股票代码 {symbol} 的全方位综合量化雷达因子数据，直接对其当前的投资价值给出一个总结性的“人话”判断。
        
        {info_str}
        
        任务要求：
        1. 必须一针见血，不要复述数据，只需回答数据背后暴露的本质问题（如：跌出黄金坑、明显高估有杀跌风险、资金正在撤离等）。
        2. 结合基本面的长逻辑与技术面的短信号，找出它们的共振点或矛盾点。
        3. 直接给出一到两句具体的交易行动命令建议。
        4. 请控制在 250 字以内，字字珠玑，格式采用Markdown。
        """

    def _build_prompt(self, symbol: str, data: dict) -> str:
        return f"""
        你是一位拥有20年经验的A股金融分析师。请根据以下关于股票代码 {symbol} 的数据生成一份专业的日间投资分析简报。
        
        【市场数据】
        - 当前收盘价: {data.get('current_price', 'N/A')}
        - AI模型预测趋势: {data.get('predicted_trend', 'Unknown')} (UP=看涨, DOWN=看跌)
        - AI模型置信度: {data.get('confidence', 0):.2f} (分值越高越确信)
        - 技术指标概览: 结合了均线(MA)、MACD动能及RSI相对强弱指数。
        
        【分析要求】
        1. **行情解读**: 解读当前价格水平及模型预测方向。
        2. **风险提示**: 针对A股市场的波动性给出理性的风险警示。
        3. **操作建议**: 基于预测趋势，给出建议（如：轻仓观察、逢低吸纳、注意止盈等）。
        
        请用中文回答，格式清晰，条理分明，字数控制在300字以内。
        """

class OllamaLLM(BaseLLM):
    def __init__(self, model_name="qwen3:1.7b"):
        # 默认尝试使用 qwen2.5, 用户需自行 pull
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
    def generate_report(self, symbol: str, data: dict) -> str:
        prompt = self._build_prompt(symbol, data)
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            # 设置较短超时，避免阻塞太久
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response field in Ollama output")
        except requests.exceptions.ConnectionError:
            return "错误：无法连接到本地 Ollama 服务。请确保 Ollama 已启动 (localhost:11434)。"
        except Exception as e:
            return f"Error generating report: {str(e)}"

    def generate_backtest_report(self, symbol: str, strategy: str, metrics: dict) -> str:
        prompt = self._build_backtest_prompt(symbol, strategy, metrics)
        print(f"DEBUG: Generating backtest report for {symbol} using {self.model_name}")
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            print(f"DEBUG: Sending request to Ollama: {self.api_url}")
            response = requests.post(self.api_url, json=payload, timeout=90) 
            response.raise_for_status()
            result = response.json()
            analysis = result.get("response", "Error: No response field in Ollama output")
            print(f"DEBUG: Ollama response received metadata: length={len(analysis)}")
            return analysis
        except requests.exceptions.ConnectionError:
            return "⚠️ **Ollama 未启动或拒绝连接**\n无法连接到本地大模型服务 (localhost:11434)。"
        except Exception as e:
            error_msg = f"Error generating backtest report: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg

    def generate_portfolio_risk_report(self, portfolio: list, risk_metrics: dict) -> str:
        prompt = self._build_portfolio_risk_prompt(portfolio, risk_metrics)
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            # 为了防止前端卡死，将风控诊断超时设为较短时间 (15s)
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response field in Ollama output")
        except requests.exceptions.ConnectionError:
            return "⚠️ **AI 风控引擎无法连接**\n无法连接到本地大模型服务 (localhost:11434)。请确保 Ollama 已启动，或前往设置切换为您指定的外部 API.\n\n(但底层量化指标已完成计算并呈现在上方)"
        except requests.exceptions.Timeout:
            return "⚠️ **AI 通讯超时**\n本地大模型正在加载或负载过高，未能及时生成风控报告。但各项精确数学指标与全球资产雷达均已计算完毕，请参考上方数据面板。"
        except Exception as e:
            return f"⚠️ **AI 风控引擎暂时离线**: {str(e)}\n\n(但底层量化指标已完成计算并呈现在上方)"

    def generate_factor_report_stream(self, symbol: str, factors_data: dict):
        prompt = self._build_factor_prompt(symbol, factors_data)
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True # 开启流式机制
            }
            with requests.post(self.api_url, json=payload, stream=True, timeout=10) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
        except requests.exceptions.Timeout:
            yield "⚠️ 抱歉，连接本地大语言模型超时，请检查模型服务是否已启动或正在加载。"
        except Exception as e:
            yield f"⚠️ 大语言模型处理异常: {str(e)}"

    def stream_generate(self, prompt: str):
        """通用的底层按词元流式生成器，供各类定制 Agent 使用"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True # 开启流式机制
            }
            with requests.post(self.api_url, json=payload, stream=True, timeout=15) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
        except requests.exceptions.Timeout:
            yield "\n**[系统警告: 大模型心跳超时，加载或推演受阻]**"
        except Exception as e:
            yield f"\n**[智能体核心错误: {str(e)}]**"

    def chat_with_tools(self, messages: list, tools: list = None):
        """
        供高级 Agent (ReAct) 调用的 /api/chat 接口。
        支持上下文 (messages) 和可用工具映射 (tools)。
        返回值形如: {"message": {"role": "assistant", "content": "...", "tool_calls": [...]}}
        该方法采用阻塞同步调用而非推流，因为工具挂载时流式处理较差。
        """
        chat_url = self.api_url.replace("/api/generate", "/api/chat")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        if tools:
            payload["tools"] = tools
            
        try:
            # 开启加长 Timeout，因为带工具池的推理常常需要深入思考
            response = requests.post(chat_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {})
        except requests.exceptions.ConnectionError:
            return {"role": "assistant", "content": "⚠️ **Ollama 未启动或拒绝连接**\n无法连接到本地大模型服务 (localhost:11434)。请确保 Ollama 守护线程已拉起，或者前往 **设置中心** 切换为您指定的外部 API 提供者 (如 DeepSeek)。"}
        except requests.exceptions.Timeout:
            return {"role": "assistant", "content": "⚠️ **节点推理超时**\n使用本地大模型加载或推演时耗时过久被熔断。"}
        except Exception as e:
            return {"role": "assistant", "content": f"⚠️ **内部通信总线崩塌**: {e}"}



class ValidationLLM(BaseLLM):
    """用于测试或无模型环境的 Mock LLM"""
    def generate_report(self, symbol: str, data: dict) -> str:
        trend = "上涨" if data.get('predicted_trend') == "UP" else "下跌"
        return f"""
        【模拟AI分析报告】
        股票代码：{symbol}
        根据混合模型预测，该股票短期内趋势看{trend}。
        当前价格为 {data.get('current_price')}。
        
        注：这是测试模式生成的报告，请在本地安装 Ollama 并下载模型以获得真实AI分析。
        """

    def generate_backtest_report(self, symbol: str, strategy: str, metrics: dict) -> str:
        return f"""
        【模拟AI回测评价】
        股票：{symbol} | 策略：{strategy}
        回测表现分析：
        - 总收益：{metrics.get('total_return_pct', 0):.2f}%
        - 最大回撤：{metrics.get('max_drawdown_pct', 0):.2f}%
        收益表现{"理想" if metrics.get('total_return_pct', 0) > 0 else "欠佳"}，最大回撤控制在 {metrics.get('max_drawdown_pct', 0):.2f}%。
        建议：继续观察市场动态。
        """

    def generate_portfolio_risk_report(self, portfolio: list, risk_metrics: dict) -> str:
        return f"""
        【模拟AI投资组合风控报告】
        评估对象包括：{len(portfolio)} 支股票
        根据计算，该组合年化波动率为 {risk_metrics.get('annual_volatility', 0)*100:.2f}%。
        99% VaR 指标预估每日极限亏损幅度约为 {risk_metrics.get('historical', {}).get('var_99', 0)*100:.2f}%。
        
        *系统级提示：请在本地 Olllama 服务启动，以获取完整而深度的真实风险解读。*
        """

    def generate_factor_report_stream(self, symbol: str, factors_data: dict):
        import time
        mock_text = f"【模拟数据：{symbol}因子投资兵推】\n\n系统目前未接入 Ollama 模型。根据传参的基本面 PE {factors_data.get('fundamental',{}).get('PE')} 以及 RSI 指数 {factors_data.get('technical',{}).get('RSI')}，此股票短期表现出特定的量化特征。\n\n**建议**：结合自身风险偏好继续观察。（请接入真实大模型环境以获取深度推演）"
        for char in mock_text:
            yield char
            time.sleep(0.01)

def get_llm_client(provider="ollama", model_name="qwen3:1.7b", api_key=None, base_url=None):
    """
    工厂函数获取 LLM 客户端
    """
    if provider == "ollama":
        return OllamaLLM(model_name=model_name)
    elif provider == "openai":
        return OpenAILLM(model_name=model_name, api_key=api_key, base_url=base_url)
    elif provider == "validation":
        return ValidationLLM()
    else:
        # 默认为 validation 以防报错
        return ValidationLLM()

class OpenAILLM(BaseLLM):
    """
    OpenAI 官方兼容网关（也适用于 DeepSeek, 零一万物大模型等支持标准 /v1/chat/completions 的外部 API）
    """
    def __init__(self, model_name="gpt-4o", api_key=None, base_url=None):
        from backend.config import settings
        from openai import OpenAI
        self.model_name = model_name
        
        # 优先使用传入的临时参数（用于前端连通性测试），否则 fallback 到配置环境变量
        actual_base_url = base_url if base_url is not None else settings.OPENAI_BASE_URL
        actual_api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        
        if not actual_api_key:
            raise ValueError("未在环境配置(.env)中找到 OPENAI_API_KEY 或对应的大模型秘钥。请先设置。")
            
        # 若有指定代理网流则加载
        if actual_base_url:
            self.client = OpenAI(api_key=actual_api_key, base_url=actual_base_url)
        else:
            self.client = OpenAI(api_key=actual_api_key)

    def generate_report(self, symbol: str, data: dict) -> str:
        prompt = self._build_prompt(symbol, data) # 复用既有提示词
        return self._sync_chat(prompt)

    def generate_backtest_report(self, symbol: str, strategy: str, metrics: dict) -> str:
        prompt = self._build_backtest_prompt(symbol, strategy, metrics)
        return self._sync_chat(prompt)

    def generate_portfolio_risk_report(self, portfolio: list, risk_metrics: dict) -> str:
        prompt = self._build_portfolio_risk_prompt(portfolio, risk_metrics)
        return self._sync_chat(prompt)

    def _sync_chat(self, prompt: str) -> str:
        """统一下挂阻塞方法"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return response.choices[0].message.content or "Error: 空响应体"
        except Exception as e:
            return f"⚠️ 外部 API 调用熔断: {str(e)}"

    def generate_factor_report_stream(self, symbol: str, factors_data: dict):
        prompt = self._build_factor_prompt(symbol, factors_data)
        yield from self.stream_generate(prompt)

    def stream_generate(self, prompt: str):
        try:
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            yield f"\n**[智能网关熔断异常: {str(e)}]**"

    def chat_with_tools(self, messages: list, tools: list = None):
        """
        供高级 Agent (ReAct) MCP 辩论引擎使用的核心方法。
        完全挂载 OpenAI Native Tool Calling 协议。
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            
        try:
            # 这也是阻塞拉取
            response = self.client.chat.completions.create(**payload, timeout=60)
            message_obj = response.choices[0].message
            
            # 手工结构化包裹为类似 Ollama 的字典返回
            ret = {
                "role": message_obj.role,
                "content": message_obj.content or ""
            }
            if message_obj.tool_calls:
                formatted_calls = []
                for tc in message_obj.tool_calls:
                    formatted_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                ret["tool_calls"] = formatted_calls
            return ret
            
        except Exception as e:
            return {"role": "assistant", "content": f"[外部大脑短路: {e}]"}

    # --- 继承各类打针 Prompt 生成方法（因在 OllamaLLM 中已经抽离所以这里只需要将原本的搬借即可，鉴于 Python 类方法查找这里我们需要从原有的抽象剥离开，为了省事，将 _build_xxxx 方法提到 BaseLLM） 

