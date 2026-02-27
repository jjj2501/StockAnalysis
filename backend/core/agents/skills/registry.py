"""
专业技能注册总表 (Skills Registry)
每个 Agent 角色可按需注册专属 MCP Tool Schema，
挂载到各自的 ReActAgent 实例中。
"""

from typing import List, Dict

# ───────────────────────────────────────────────
# 宏观分析师专属技能
# ───────────────────────────────────────────────
MACRO_SKILLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_macro_data",
            "description": "获取目标资产所在市场的宏观经济指标（LPR、CPI、PMI、非农等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票/资产代码，用于推断所属市场"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_yield_curve",
            "description": "分析当前利率收益率曲线形状（正斜率/倒挂），判断衰退信号",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票/资产代码"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]

# ───────────────────────────────────────────────
# 量化研究员专属技能
# ───────────────────────────────────────────────
QUANT_SKILLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_technical_data",
            "description": "获取技术指标（RSI, MACD, MA均线, 布林带, 成交量背离等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamental_data",
            "description": "获取基本面数据（PE/PB/ROE/营收增速/净利润，对标行业均值）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_reports",
            "description": "在财报/电话会议纪要/公告向量库中检索特定主题的原文段落（如：利润下降原因、扩产计划）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"},
                    "query": {"type": "string", "description": "检索主题描述，如'毛利率下降原因'"}
                },
                "required": ["symbol", "query"]
            }
        }
    }
]

# ───────────────────────────────────────────────
# 风控官专属技能
# ───────────────────────────────────────────────
RISK_SKILLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "获取近期舆情新闻标题列表及情感得分（-100 恐慌 ~ +100 狂热），识别黑天鹅风险",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_var_95",
            "description": "计算该股票在 95% 置信水平下的在险价值（VaR），用于评估下行波动风险敞口",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"},
                    "days": {"type": "integer", "description": "计算窗口（历史天数），默认 252", "default": 252}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_data",
            "description": "获取技术指标辅助确认超买/超卖及趋势破位信号",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        }
    }
]

# 技能映射字典：角色名 → 技能列表
ROLE_SKILLS = {
    "Macro Analyst": MACRO_SKILLS,
    "Quant Researcher": QUANT_SKILLS,
    "Risk Control Agent": RISK_SKILLS,
}
