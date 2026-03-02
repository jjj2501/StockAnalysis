"""
因子 IC（信息系数）分析引擎
计算技术因子对未来收益的预测能力，输出 IC、ICIR、IC 衰减曲线与因子相关矩阵。
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats

logger = logging.getLogger(__name__)

# 参与 IC 分析的因子列表（列名 -> 中文名）
FACTOR_META = {
    "RSI":        "RSI 相对强弱",
    "MACD_Hist":  "MACD 柱量",
    "WR":         "威廉指标",
    "ROC":        "12日变动率",
    "ATR_Norm":   "标准化波幅",
    "BB_Pos":     "布林带位置",
    "Vol_Ratio":  "量能比（量/20日均量）",
    "Mom5":       "5日价格动量",
    "Mom20":      "20日价格动量",
}

# 预测期（日）
FORWARD_DAYS = [1, 3, 5, 10, 20]


class FactorICAnalyzer:
    """
    因子 IC 分析器。
    基于本地 Parquet K 线历史数据，构建因子时序，计算各因子的 IC/ICIR/衰减曲线。
    """

    def compute_factor_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从 OHLCV DataFrame 计算所有因子的历史时序，追加到 df 中并返回。
        """
        if df.empty or len(df) < 30:
            return df

        c = df["close"]
        v = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        # ── 已有指标（直接复用 data.py 的计算逻辑）──
        # RSI 14
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD Histogram
        exp1 = c.ewm(span=12, adjust=False).mean()
        exp2 = c.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = macd - signal

        # 威廉指标 WR
        hh = df["high"].rolling(14).max()
        ll = df["low"].rolling(14).min()
        df["WR"] = -100 * (hh - c) / (hh - ll + 1e-9)

        # ROC 12日变动率
        df["ROC"] = c.pct_change(12) * 100

        # ATR_Norm：ATR / 收盘价（标准化，消除绝对量级差异）
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - c.shift(1)).abs()
        tr3 = (df["low"] - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["ATR_Norm"] = atr / (c + 1e-9) * 100

        # BB 位置：(收盘 - 下轨) / (上轨 - 下轨)，0=下轨，1=上轨
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["BB_Pos"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # 量能比
        avg_vol = v.rolling(20).mean()
        df["Vol_Ratio"] = v / (avg_vol + 1e-9)

        # 价格动量
        df["Mom5"] = c.pct_change(5) * 100
        df["Mom20"] = c.pct_change(20) * 100

        return df

    def compute_forward_returns(self, df: pd.DataFrame, n: int) -> pd.Series:
        """计算 N 日后的对数收益率"""
        return np.log(df["close"].shift(-n) / df["close"])

    def compute_ic_series(
        self,
        factor: pd.Series,
        fwd_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        计算滚动窗口内的 Spearman IC 序列。
        IC = rank_corr(factor_t, fwd_return_t) 在每个窗口内计算。
        """
        # 对齐并去掉 NaN 后再滚动
        combined = pd.concat([factor, fwd_returns], axis=1).dropna()
        if len(combined) < window:
            return pd.Series(dtype=float)

        factor_clean = combined.iloc[:, 0]
        ret_clean = combined.iloc[:, 1]

        ic_vals = []
        dates = []
        for i in range(window, len(combined)):
            f_win = factor_clean.iloc[i - window:i]
            r_win = ret_clean.iloc[i - window:i]
            ic, _ = stats.spearmanr(f_win, r_win)
            ic_vals.append(ic if not np.isnan(ic) else 0.0)
            dates.append(combined.index[i])

        return pd.Series(ic_vals, index=dates)

    def compute_icir(self, ic_series: pd.Series) -> Dict:
        """计算 IC 均值、标准差、ICIR、IC>0比率"""
        if ic_series.empty:
            return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "ic_positive_rate": 0.0}

        clean = ic_series.dropna()
        ic_mean = float(clean.mean())
        ic_std = float(clean.std()) if len(clean) > 1 else 1e-9
        icir = ic_mean / (ic_std + 1e-9)
        positive_rate = float((clean > 0).mean())

        return {
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "icir": round(icir, 4),
            "ic_positive_rate": round(positive_rate, 4),
        }

    def compute_ic_decay(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        forward_days: List[int] = FORWARD_DAYS,
        min_obs: int = 50
    ) -> Dict[str, List]:
        """
        计算各因子在不同预测期（1/3/5/10/20日）的 IC，返回衰减表。

        返回格式：
        {
            "forward_days": [1, 3, 5, 10, 20],
            "factors": {
                "RSI": [ic_1d, ic_3d, ic_5d, ic_10d, ic_20d],
                ...
            }
        }
        """
        result = {"forward_days": forward_days, "factors": {}}

        for col in factor_cols:
            if col not in df.columns:
                continue
            ic_per_day = []
            for n in forward_days:
                fwd = self.compute_forward_returns(df, n)
                combined = pd.concat([df[col], fwd], axis=1).dropna()
                if len(combined) < min_obs:
                    ic_per_day.append(0.0)
                    continue
                ic, _ = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
                ic_per_day.append(round(float(ic) if not np.isnan(ic) else 0.0, 4))
            result["factors"][col] = ic_per_day

        return result

    def compute_factor_corr(
        self, df: pd.DataFrame, factor_cols: List[str]
    ) -> Dict:
        """计算因子间的 Spearman 相关矩阵"""
        valid_cols = [c for c in factor_cols if c in df.columns]
        if len(valid_cols) < 2:
            return {"labels": valid_cols, "matrix": []}

        sub = df[valid_cols].dropna()
        if len(sub) < 30:
            return {"labels": valid_cols, "matrix": []}

        corr_matrix = sub.corr(method="spearman")
        return {
            "labels": valid_cols,
            "matrix": [[round(float(v), 3) for v in row]
                       for row in corr_matrix.values],
        }

    def run_full_analysis(
        self,
        symbol: str,
        market: str = "CN",
        lookback_days: int = 500,
        ic_window: int = 60,
    ) -> Dict:
        """
        完整 IC 分析入口。
        1. 加载历史 K 线数据
        2. 计算因子时序
        3. 返回 IC 汇总表、IC 衰减曲线、因子相关矩阵
        """
        from backend.core.data import DataFetcher
        import datetime
        fetcher = DataFetcher()

        # 计算 start_date / end_date（YYYYMMDD 格式）
        today = datetime.datetime.now()
        start_dt = today - datetime.timedelta(days=lookback_days * 1.5)  # 乘以 1.5 保障足够数据
        start_date = start_dt.strftime("%Y%m%d")
        end_date = today.strftime("%Y%m%d")

        # 加载历史数据
        df = fetcher.get_stock_data(symbol, start_date=start_date, end_date=end_date, market=market)
        if df is None or df.empty or len(df) < 100:
            return {"error": f"历史数据不足（仅 {len(df) if df is not None else 0} 条），无法进行 IC 分析。请确保该股票有足够的历史缓存。"}

        # 计算因子历史
        df = self.compute_factor_history(df.copy())
        factor_cols = list(FACTOR_META.keys())

        # ── IC 汇总表 ──
        fwd_1d = self.compute_forward_returns(df, 1)
        summary = []
        for col in factor_cols:
            if col not in df.columns:
                continue
            ic_s = self.compute_ic_series(df[col], fwd_1d, window=ic_window)
            stats_dict = self.compute_icir(ic_s)
            ic_mean = stats_dict["ic_mean"]
            icir = stats_dict["icir"]
            # 建议标签
            if abs(icir) >= 0.5 and abs(ic_mean) >= 0.03:
                suggestion = "✅ 推荐使用"
            elif abs(ic_mean) >= 0.02:
                suggestion = "⚠️ 参考使用"
            else:
                suggestion = "❌ 预测力弱"

            summary.append({
                "factor": col,
                "name": FACTOR_META.get(col, col),
                **stats_dict,
                "suggestion": suggestion,
            })

        # 按 |ICIR| 排序
        summary.sort(key=lambda x: abs(x["icir"]), reverse=True)

        # ── IC 衰减曲线 ──
        decay = self.compute_ic_decay(df, factor_cols)

        # ── 因子相关热力图 ──
        corr = self.compute_factor_corr(df, factor_cols)

        return {
            "symbol": symbol,
            "market": market,
            "total_bars": len(df),
            "factor_summary": summary,
            "ic_decay": decay,
            "factor_corr": corr,
        }
