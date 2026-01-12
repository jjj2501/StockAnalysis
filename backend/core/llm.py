import abc
import requests
import json
import os
from typing import Optional

class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate_report(self, symbol: str, data: dict) -> str:
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
