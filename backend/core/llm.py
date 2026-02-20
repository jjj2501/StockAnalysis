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
            response = requests.post(self.api_url, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response field in Ollama output")
        except Exception as e:
            return f"Error generating portfolio risk report: {str(e)}"

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
        # 提取组合构成字符串
        portfolio_str = ", ".join([f"{p['symbol']}({p.get('shares', 0)}股)" for p in portfolio])
        
        metrics = risk_metrics.get("metrics", {})
        hist = metrics.get("historical", {})
        perf = risk_metrics.get("performance", {})
        corr = risk_metrics.get("correlation_matrix", {})
        
        # 简化相关性矩阵的展示，提取核心数据给 LLM
        corr_str = json.dumps(corr, ensure_ascii=False)
        
        return f"""
        你是一位极其严苛的机构级量化风控官。请根据以下个人投资组合的真实历史回测数据，用“无情且极其直白”的大白话，撰写一份专业的风控诊断及调仓药方报告。你的目标是打醒那些盲目自信的散户。
        
        【投资组合构成】
        - 包含资产: {portfolio_str}
        - 组合总价值: {risk_metrics.get('total_value', 0):.2f} 元
        - 组合年化波动率: {risk_metrics.get('annual_volatility', 0)*100:.2f}%
        
        【机构核心绩效与回撤指标】
        - 过去一年真实最大回撤幅度: {perf.get('max_drawdown_pct', 0)*100:.2f}% 
        - 过去一年真实最大亏损金额: {perf.get('max_drawdown_amount', 0):.2f} 元
        - 夏普比率 (Sharpe Ratio): {perf.get('sharpe_ratio', 0):.2f} (大于1算优秀，负数说明连无风险收益都不如)
        - 索提诺比率 (Sortino Ratio): {perf.get('sortino_ratio', 0):.2f}
        
        【内部资产相关性矩阵】(数值越接近1代表同涨同跌，缺乏分散性)
        {corr_str}
        
        【风险价值 (VaR) 尾部风险】
        - 99% 置信度单日极限回撤预估 (VaR 99%): {hist.get('var_99', 0)*100:.2f}%
        - 黑天鹅事件极端单日跌幅预估 (CVaR 99%): {hist.get('cvar_99', 0)*100:.2f}%
        
        【分析要求】
        1. **“能不能拿得住” (最大回撤痛点分析)**：结合绝对亏损金额，直击灵魂地质问这套组合在最惨时跌掉的钱是否超出了普通人的心理承受极限。
        2. **“这是不是瞎折腾” (夏普比率绩效点评)**：评价其夏普比率。如果极低，指出承担了高风险却没换来收益。
        3. **“假分散还是真集中” (相关性雷达)**：根据皮尔逊相关性矩阵数据，揪出哪些股票是“穿同一条裤子”的高度相关标的，警告这不是真正的分散。
        4. **【核心】粗暴直接的调仓药方**: 作为机构风控官，直接给出具体“手术级别”的对冲或调仓建议（如：削减某两只高相关性仓位，配置低相关性防守资产或保留现金）。

        请用 Markdown 格式，语气务必专业、犀利、直击要害，字数控制在 500 字以内。
        """

    def _build_prompt(self, symbol: str, data: dict) -> str:
        # 构建提示词
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

def get_llm_client(provider="ollama", model_name="qwen3:1.7b"):
    """
    工厂函数获取 LLM 客户端
    """
    if provider == "ollama":
        return OllamaLLM(model_name=model_name)
    elif provider == "validation":
        return ValidationLLM()
    else:
        # 默认为 validation 以防报错
        return ValidationLLM()
