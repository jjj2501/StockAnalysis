import json
import time
from functools import wraps
from backend.core.data import DataFetcher

# 创建一个常驻单例以供各类函数共用，节省拉起速度
_shared_fetcher = DataFetcher()

def ttl_cache(ttl_seconds=300):
    """
    轻量级的本地内存级防抖缓存装饰器。
    解决多智能体在同一回合内针对同一只股票高频查询同一指标，导致触发防爬虫以及拖慢响应的问题。
    """
    cache = {}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 将传参转换为可哈希的 key
            key = str(args) + str(kwargs)
            now = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
            # 缓存穿透或过期，引发真实调用
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

@ttl_cache(ttl_seconds=300)
def get_macro_data(symbol: str) -> str:
    """获取标的资产所处市场的宏观经济指标（如LPR, PMI, CPI等）"""
    clean_symbol, market = _shared_fetcher.parse_symbol_market(symbol)
    try:
        data = _shared_fetcher._fetch_macro(market)
        if not data or "macro" not in data:
            return f'{{"macro": {{"notice": "宏观数据源暂时连接受限，请宏观分析师直接根据已有市场常识或其他数据线索继续推演，严格禁止再次调用该工具重试！"}}}}'
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f'{{"macro": {{"notice": "宏观数据源连接报错，请宏观分析师直接基于已有信息汇总，严格禁止再次调用该工具重试！"}}}}'

@ttl_cache(ttl_seconds=300)
def get_technical_data(symbol: str) -> str:
    """获取标的资产近期最核心的技术分析与量价指标（如RSI, MACD, MA均线以及动量偏离）"""
    try:
        data = _shared_fetcher._fetch_technical(symbol)
        if not data or ("technical" not in data and "error" not in data):
            return f'{{"technical": {{"notice": "技术指标获取熔断，请假设各项指标处于中性水平并继续执行下一步计划，严格禁止重试！"}}}}'
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f'{{"technical": {{"notice": "技术指标获取报错，请假设技术趋势不明朗并继续执行您的最后总结，严格禁止重试！"}}}}'

@ttl_cache(ttl_seconds=300)
def get_fundamental_data(symbol: str) -> str:
    """获取该公司的财务体检表指标，包含市盈率(PE)、市净率(PB)与近期营收表现"""
    try:
        clean_symbol, market = _shared_fetcher.parse_symbol_market(symbol)
        data = _shared_fetcher._fetch_fundamental(clean_symbol, market=market, original_symbol=symbol)
        if not data or "fundamental" not in data:
            return f'{{"fundamental": {{"notice": "财务体检表暂无返回，请依靠您的预备知识或其他探针数据继续推力，绝不准再次重试！"}}}}'
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f'{{"fundamental": {{"notice": "财务体检表获取报错，请停止调查此项并基于其他特征得出您的买卖结论，严格禁止重试！"}}}}'

@ttl_cache(ttl_seconds=300)
def get_news_sentiment(symbol: str) -> str:
    """调用新闻与研报舆情爬虫，查看该资产近期被定性为利空还是利好，以及情绪打分"""
    try:
        data = _shared_fetcher._fetch_news_sentiment(symbol)
        if not data or "news" not in data:
            return f'{{"news": {{"notice": "舆情探针受阻，请视作近期无重大基本面新闻发生并继续您的沙盘推演，绝不准重试！"}}}}'
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f'{{"news": {{"notice": "舆情探针报错，请直接继续当前推演逻辑走向结论收尾，严格限制重试动作！"}}}}'


# ==========================================
# Ollama / OpenAI 标准 Tools (MCP) 定义字典
# ==========================================
SYSTEM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_macro_data",
            "description": "获取股票/资产所属大盘/国家的重要宏观经济数据(失业率/利率/PMI)以研判国家队方向和水泵流动性。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "资产代码，例如 'AAPL.O' 或 '600519'"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_data",
            "description": "获取资产短线上最重要的技术切线形态（含是否超买、均线发散程度、背离等），供交易员寻找低吸或逃顶位。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "资产代码"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamental_data",
            "description": "获取资产最新的财务估值，用于判定当前股价究竟是存在泡沫还是被严重错杀低估的黄金坑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "资产代码"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "爬取有关该资产的最新舆情动向，获取打分。用于排雷潜在的黑天鹅恶劣事件或确认概念炒作的温度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "资产代码"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]

# 用于被调用时动态映射的函数注册表
TOOL_REGISTRY = {
    "get_macro_data": get_macro_data,
    "get_technical_data": get_technical_data,
    "get_fundamental_data": get_fundamental_data,
    "get_news_sentiment": get_news_sentiment
}
