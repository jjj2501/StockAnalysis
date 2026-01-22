import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """策略基类"""
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        :param df: 包含必要指标的 DataFrame
        :return: 信号序列 (1: 买入, -1: 卖出, 0: 持仓/观望)
        """
        pass

class MACDStrategy(Strategy):
    """基于 MACD 的金叉/死叉策略"""
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))] = 1
        signals[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))] = -1
        return signals

class RSIStrategy(Strategy):
    """基于 RSI 的超买超卖策略"""
    def __init__(self, low=30, high=70):
        self.low = low
        self.high = high

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df['RSI'] < self.low] = 1
        signals[df['RSI'] > self.high] = -1
        return signals

class AIStrategy(Strategy):
    """
    向量化优化的 AI 预测策略
    """
    def __init__(self, engine, symbol: str):
        self.engine = engine
        self.symbol = symbol

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        利用批量推理提升 AI 策略回测性能
        """
        logger.info(f"开始为 {self.symbol} 执行批量 AI 信号生成...")
        signals = pd.Series(0, index=df.index)
        
        try:
            # 调用 engine 的批量推理接口
            predictions = self.engine.predict_batch(self.symbol, df)
            for i, pred in enumerate(predictions):
                if pred == "UP": signals.iloc[i] = 1
                elif pred == "DOWN": signals.iloc[i] = -1
        except Exception as e:
            logger.error(f"批量 AI 推理失败: {e}, 退回到基础逻辑")
            signals[(df['RSI'] < 35)] = 1
            signals[(df['RSI'] > 65)] = -1
            
        return signals

class BacktestEngine:
    """
    回测引擎 (支持循环与向量化模式)
    """
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.0003):
        self.initial_cash = initial_cash
        self.commission = commission

    def run_vectorized(self, df: pd.DataFrame, strategy: Strategy) -> Dict[str, Any]:
        """
        高性能回测 (稀疏事件驱动 + 向量化计算)
        兼顾了回测速度与交易细节的准确性
        """
        if df.empty:
            return {"error": "Empty data"}

        df = df.copy()
        signals = strategy.generate_signals(df)
        
        # 识别持仓变化点 (交易点)
        # 1: 买入信号, -1: 卖出信号
        # 注意: 这里的 signal 代表"目标持仓状态"还是"买卖动作"? 
        # Strategy 定义: 1: 买入/持有, -1: 卖出/空仓, 0: 观望
        # 我们假设:
        # signal=1 且 curr_pos=0 -> BUY
        # signal=-1 且 curr_pos>0 -> SELL
        
        # 将信号转换为目标持仓 (0 或 1)
        target_pos = signals.replace(0, np.nan).ffill().fillna(0)
        target_pos = target_pos.replace(-1, 0) # 暂不支持做空
        
        # 找出持仓发生变化的时刻
        pos_diff = target_pos.diff().fillna(0)
        if target_pos.iloc[0] == 1:
            pos_diff.iloc[0] = 1 # 第一天即买入
            
        trade_indices = df.index[pos_diff != 0]
        
        trades = []
        cash = self.initial_cash
        shares = 0
        
        # 记录持仓状态变更，用于后续向量化计算净值
        # 格式: date_idx -> (shares, cash)
        state_changes = {}
        # 初始状态
        state_changes[df.index[0]] = (0, self.initial_cash)
        
        if len(trade_indices) > 0:
            # 稀疏迭代: 仅遍历交易日 (性能 O(TradeCount))
            for idx in trade_indices:
                row = df.loc[idx]
                price = float(row['close'])
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                change = pos_diff.loc[idx]
                
                if change == 1 and cash > 0: # BUY
                    # 全仓买入
                    buy_shares = int(cash / (price * (1 + self.commission)))
                    if buy_shares > 0:
                        cost = buy_shares * price * (1 + self.commission)
                        cash -= cost
                        shares += buy_shares
                        trades.append({
                            "date": date_str, "type": "BUY", "price": price, 
                            "shares": buy_shares, "cost": float(f"{cost:.2f}")
                        })
                elif change == -1 and shares > 0: # SELL
                    # 清仓卖出
                    revenue = shares * price * (1 - self.commission)
                    cash += revenue
                    trades.append({
                        "date": date_str, "type": "SELL", "price": price, 
                        "shares": shares, "revenue": float(f"{revenue:.2f}")
                    })
                    shares = 0
                    
                # 记录该日收盘后的状态
                state_changes[idx] = (shares, cash)

        # --- 向量化计算资金曲线 (性能 O(N)) ---
        # 1. 构建状态 DataFrame
        states_df = pd.DataFrame.from_dict(state_changes, orient='index', columns=['shares', 'cash'])
        # 2. 重新索引到全量日期并填充
        states_df = states_df.reindex(df.index).ffill()
        # 确保初始状态填充 (如果第一天没有交易)
        states_df['shares'] = states_df['shares'].fillna(0)
        states_df['cash'] = states_df['cash'].fillna(self.initial_cash)
        
        # 3. 计算每日净值 = 持仓市值 + 现金
        equity_series = states_df['shares'] * df['close'] + states_df['cash']
        
        equity_curve = []
        # Optimization: Use to_dict only once or format manually to speed up
        dates = df['date'].dt.strftime('%Y-%m-%d')
        equity_values = equity_series.values
        
        equity_curve = [{"date": d, "equity": float(v)} for d, v in zip(dates, equity_values)]

        stats = self.calc_metrics(equity_curve)
        return {
            "summary": stats,
            "equity_curve": equity_curve,
            "trades": trades,
            "mode": "vectorized_optimized"
        }

    def run(self, df: pd.DataFrame, strategy: Strategy) -> Dict[str, Any]:
        """封装默认回测模式"""
        # 对于不需要精细模拟持仓量的策略，默认使用向量化
        return self.run_vectorized(df, strategy)

    def _run_loop(self, df: pd.DataFrame, strategy: Strategy) -> Dict[str, Any]:
        """传统的循环回测逻辑"""
        signals = strategy.generate_signals(df)
        cash = self.initial_cash
        position = 0 
        equity_curve = []
        trades = []
        
        for i in range(len(df)):
            sig = signals.iloc[i]
            price = df['close'].iloc[i]
            date_str = df['date'].iloc[i].strftime('%Y-%m-%d') if hasattr(df['date'].iloc[i], 'strftime') else str(df['date'].iloc[i])
            
            if sig == 1 and position == 0:
                shares = int(cash / (price * (1 + self.commission)))
                if shares > 0:
                    cost = shares * price * (1 + self.commission)
                    cash -= cost
                    position = shares
                    trades.append({"date": date_str, "type": "BUY", "price": float(price), "shares": shares, "cost": float(cost)})
            elif sig == -1 and position > 0:
                revenue = position * price * (1 - self.commission)
                cash += revenue
                trades.append({"date": date_str, "type": "SELL", "price": float(price), "shares": position, "revenue": float(revenue)})
                position = 0
            
            total_equity = cash + position * price
            equity_curve.append({"date": date_str, "equity": float(total_equity)})
            
        stats = self.calc_metrics(equity_curve)
        return {"summary": stats, "equity_curve": equity_curve, "trades": trades, "mode": "loop"}

    def calc_metrics(self, equity_curve: List[Dict]) -> Dict[str, Any]:
        """计算绩效指标"""
        if not equity_curve:
            return {}
            
        equities = [item['equity'] for item in equity_curve]
        equities_series = pd.Series(equities)
        
        total_return = (equities[-1] - equities[0]) / equities[0]
        daily_returns = equities_series.pct_change().dropna()
        
        num_days = len(equities)
        annual_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0
            
        roll_max = equities_series.cummax()
        drawdown = (equities_series - roll_max) / roll_max
        max_drawdown = drawdown.min()
        
        risk_free = 0.03 / 252
        if daily_returns.std() != 0:
            sharpe = (daily_returns.mean() - risk_free) / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
            
        return {
            "initial_cash": self.initial_cash,
            "final_equity": float(equities[-1]),
            "total_return_pct": float(total_return * 100),
            "annual_return_pct": float(annual_return * 100),
            "max_drawdown_pct": float(max_drawdown * 100),
            "sharpe_ratio": float(sharpe),
            "trade_days": num_days
        }
