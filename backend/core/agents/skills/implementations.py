"""
技能函数实现层：compute_var_95 等补充技能的实际执行逻辑
其余技能（get_technical_data, get_macro_data 等）复用 tools.py 中的同名函数
"""

import json
import logging
import numpy as np
from backend.core.data import DataFetcher

logger = logging.getLogger(__name__)
_fetcher = DataFetcher()


def compute_var_95(symbol: str, days: int = 252) -> str:
    """
    历史模拟法 VaR（95% 置信区间，单日持仓损失上限）
    返回 JSON 字符串，供 Agent 作为 Observation 消费。
    """
    try:
        # 先解析市场代码
        try:
            clean_sym, market = _fetcher.parse_symbol_market(symbol)
        except Exception:
            clean_sym, market = symbol, "US"

        df = _fetcher.get_stock_data(clean_sym, market=market, days=days + 10)
        if df is None or df.empty or len(df) < 30:
            return json.dumps({"notice": "数据不足，无法计算 VaR"}, ensure_ascii=False)

        # 计算日收益率
        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]

        # 历史模拟法：取最差 5% 分位
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(np.mean(returns[returns <= var_95]))  # 条件 VaR (CVaR)

        signal = "🔴 尾部风险较高" if var_95 < -0.03 else ("🟡 中等波动" if var_95 < -0.015 else "🟢 波动可控")

        return json.dumps({
            "var_95": {
                "value": round(var_95 * 100, 2),
                "unit": "%",
                "description": f"95% 置信水平下单日最大损失约 {abs(var_95)*100:.2f}%",
                "signal": signal
            },
            "cvar_95": {
                "value": round(cvar_95 * 100, 2),
                "unit": "%",
                "description": f"超出 VaR 阈值时的平均损失（条件 VaR）"
            },
            "sample_days": len(returns)
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[VaR计算] {symbol} 失败: {e}")
        return json.dumps({"notice": f"VaR 计算遭遇异常: {e}"}, ensure_ascii=False)


def analyze_yield_curve(symbol: str) -> str:
    """
    简化版收益率曲线倒挂判断（基于宏观数据中的利率差）
    返回对曲线形态的定性描述，帮助宏观分析师判断衰退信号。
    """
    try:
        macro = _fetcher.get_macro_data(market="US")
        fed_rate = macro.get("US_Fed_Rate", {}).get("value", None)
        result = {
            "status": "数据获取受限" if fed_rate is None else "已获取",
            "interpretation": (
                "美联储基准利率较高，关注短债长债利差是否倒挂——历史上倒挂超 6 个月通常预示衰退。"
                if fed_rate and float(str(fed_rate)) > 4 else
                "利率处于中性区间，收益率曲线倒挂风险相对可控。"
            ),
            "notice": "本工具基于公开宏观数据做定性判断，建议结合 Bloomberg/Wind 实时数据验证。"
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"notice": f"收益率曲线分析暂不可用: {e}"}, ensure_ascii=False)
