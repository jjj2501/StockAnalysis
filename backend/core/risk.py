import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import datetime

from backend.core.data import DataFetcher

logger = logging.getLogger(__name__)

class RiskManager:
    """投资组合风控计算引擎"""
    
    def __init__(self):
        self.fetcher = DataFetcher()

    def calculate_portfolio_risk(self, portfolio: List[Dict[str, Any]], days_history: int = 252) -> Dict[str, Any]:
        """
        计算投资组合的总体风险指标 (VaR, CVaR 等)
        :param portfolio: [{"symbol": "600519", "shares": 100, "price": 1680.5}, ...]
        :param days_history: 使用过去多少个交易日的数据 (默认 252 即一年)
        """
        if not portfolio:
            return {"error": "投资组合为空"}
            
        try:
            # 1. 计算总市值与股票初始权重
            total_value = sum(p["shares"] * p["price"] for p in portfolio)
            if total_value == 0:
                return {"error": "投资组合总市值为0"}

            weights = {p["symbol"]: (p["shares"] * p["price"]) / total_value for p in portfolio}
            
            # 2. 获取历史数据
            end_date_str = datetime.datetime.now().strftime("%Y%m%d")
            # 考虑非交易日，多取一些日历天数以保证能拿到足够的交易日数据
            start_date_str = (datetime.datetime.now() - datetime.timedelta(days=days_history + 100)).strftime("%Y%m%d")
            
            returns_dict = {}
            for p in portfolio:
                sym = p["symbol"]
                df = self.fetcher.get_stock_data(sym, start_date=start_date_str, end_date=end_date_str)
                if df.empty:
                    logger.warning(f"无法获取股票 {sym} 的历史数据用于风控计算")
                    # 假定该股票的收益率为0
                    continue
                
                # 计算每日收益率 (使用对数或简单收益率，这里使用简单收益率 pct_change)
                df['daily_return'] = df['close'].pct_change()
                # 剔除第一个 NaN，或者直接用 tail 保证长度一致
                df = df.dropna()
                # 只取最近的 days_history 天
                df = df.tail(days_history)
                # 将日期作为 index 生成 pd.Series
                df.set_index('date', inplace=True)
                returns_dict[sym] = df['daily_return']
                
            if not returns_dict:
                return {"error": "无法获取组合内任何股票的历史数据"}

            # 将有数据的不同股票合成一个 DataFrame，日期对齐
            returns_df = pd.DataFrame(returns_dict).fillna(0) # 某只股票如果在某天停牌没有数据，假设收益为0

            # 3. 计算投资组合的历史每日整体收益率
            # portfolio_return = w1*r1 + w2*r2 + ...
            # 对于有数据的列才进行加权
            portfolio_daily_returns = pd.Series(0.0, index=returns_df.index)
            actual_weights_sum = sum(weights.get(sym, 0) for sym in returns_df.columns)
            
            if actual_weights_sum == 0:
                return {"error": "有效成分股权重为0"}

            # 归一化可用数据的权重
            normalized_weights = {sym: weights[sym] / actual_weights_sum for sym in returns_df.columns}

            for sym in returns_df.columns:
                portfolio_daily_returns += returns_df[sym] * normalized_weights[sym]

            # 4. 指标计算
            # 组合年化波动率
            daily_volatility = portfolio_daily_returns.std()
            annual_volatility = daily_volatility * np.sqrt(252)

            # --- 机构级业绩指标 (Performance Metrics) ---
            # 最大回撤 (Max Drawdown)
            cumulative_returns = (1 + portfolio_daily_returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown_pct = drawdowns.min()
            max_drawdown_amount = max_drawdown_pct * total_value
            
            # 夏普与索提诺比率 (假设无风险利率2%)
            risk_free_rate = 0.02
            daily_rf = risk_free_rate / 252
            excess_returns = portfolio_daily_returns - daily_rf
            sharpe_ratio = 0.0
            if daily_volatility > 0:
                sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_volatility
                
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = downside_returns.std(ddof=0)
            sortino_ratio = 0.0
            if downside_std > 0:
                sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std

            # 资产相关性矩阵 (Correlation Matrix)
            correlation_matrix = returns_df.corr().round(4).to_dict()

            # --- 历史模拟法 VaR (Historical VaR) ---
            # 直接找历史序列的 5% 和 1% 分位数
            hist_var_95 = np.percentile(portfolio_daily_returns, 5)
            hist_var_99 = np.percentile(portfolio_daily_returns, 1)

            # --- CVaR/Expected Shortfall (历史法) ---
            # 小于 VaR 的收益率的均值
            cvar_95 = portfolio_daily_returns[portfolio_daily_returns <= hist_var_95].mean()
            cvar_99 = portfolio_daily_returns[portfolio_daily_returns <= hist_var_99].mean()

            # --- 参数法 VaR (Parametric VaR) 假设正态分布 ---
            mean_return = portfolio_daily_returns.mean()
            from scipy.stats import norm
            param_var_95 = norm.ppf(0.05, loc=mean_return, scale=daily_volatility)
            param_var_99 = norm.ppf(0.01, loc=mean_return, scale=daily_volatility)

            # 将分布数据转换为前端易于图表化的格式 (直方图数据)
            # 为了减少数据量，我们可以将其分桶 (histogram bins)
            counts, bin_edges = np.histogram(portfolio_daily_returns * 100, bins=40, range=(-10, 10))
            histogram_data = {
                "bins": [round(b, 2) for b in bin_edges[:-1]],
                "counts": counts.tolist()
            }

            return {
                "status": "success",
                "total_value": total_value,
                "annual_volatility": float(annual_volatility),
                "daily_volatility": float(daily_volatility),
                "metrics": {
                    "historical": {
                        "var_95": float(hist_var_95),
                        "var_99": float(hist_var_99),
                        "cvar_95": float(cvar_95) if not pd.isna(cvar_95) else 0.0,
                        "cvar_99": float(cvar_99) if not pd.isna(cvar_99) else 0.0
                    },
                    "parametric": {
                        "var_95": float(param_var_95),
                        "var_99": float(param_var_99)
                    }
                },
                "performance": {
                    "max_drawdown_pct": float(max_drawdown_pct) if not pd.isna(max_drawdown_pct) else 0.0,
                    "max_drawdown_amount": float(max_drawdown_amount) if not pd.isna(max_drawdown_amount) else 0.0,
                    "sharpe_ratio": float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0.0,
                    "sortino_ratio": float(sortino_ratio) if not pd.isna(sortino_ratio) else 0.0
                },
                "correlation_matrix": correlation_matrix,
                # 可以返回一部分收益序列供前端如果需要绘制时序图
                "recent_returns": portfolio_daily_returns.tail(30).round(4).tolist(),
                "histogram": histogram_data
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
