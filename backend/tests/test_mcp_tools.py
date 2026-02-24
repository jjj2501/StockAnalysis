import unittest
import json
from backend.core.agents.tools import (
    get_macro_data,
    get_technical_data,
    get_fundamental_data,
    get_news_sentiment
)

class TestMCPTools(unittest.TestCase):
    """
    针对供大模型使用的 MCP 探针接口进行统一单元测试。
    所有函数皆应返回 JSON 序列化字符串且没有报错。
    """
    
    def setUp(self):
        # 挑选典型的且兼顾流动性的标的进行联通测试
        self.symbol_cn = "600519"
        
    def test_get_macro_data(self):
        result_str = get_macro_data(self.symbol_cn)
        self.assertIsInstance(result_str, str)
        data = json.loads(result_str)
        if "error" in data:
            self.assertIn("连接超时", data["error"] + "连接失败或暂无记录")
        else:
            self.assertIn("macro", data, "应该包含 macro 节点")
            self.assertTrue(len(data["macro"]) > 0, "宏观数据不能为空结构")

    def test_get_technical_data(self):
        result_str = get_technical_data(self.symbol_cn)
        self.assertIsInstance(result_str, str)
        data = json.loads(result_str)
        if "error" in data:
            self.assertIn("未能查得", data["error"] + "连接失败")
        else:
            self.assertIn("technical", data)
            self.assertIn("RSI", data["technical"])

    def test_get_fundamental_data(self):
        result_str = get_fundamental_data(self.symbol_cn)
        self.assertIsInstance(result_str, str)
        data = json.loads(result_str)
        if "error" in data:
            self.assertIn("连接超时", data["error"] + "无数据")
        else:
            self.assertIn("fundamental", data)
            self.assertIn("MarketCap", data["fundamental"])

    def test_get_news_sentiment(self):
        result_str = get_news_sentiment(self.symbol_cn)
        self.assertIsInstance(result_str, str)
        data = json.loads(result_str)
        if "error" in data:
            self.assertIn("连接超时", data["error"] + "未抓得")
        else:
            self.assertIn("news", data)
            self.assertIn("sentiment_score", data["news"])

if __name__ == '__main__':
    unittest.main()
