"""
因子 IC 分析引擎单元测试
"""

import pytest
import numpy as np
import pandas as pd
from backend.core.ic_analyzer import FactorICAnalyzer, FACTOR_META, FORWARD_DAYS


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成带有趋势的 mock OHLCV DataFrame"""
    rng = np.random.RandomState(seed)
    prices = 100 + np.cumsum(rng.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open":   prices + rng.rand(n) * 0.5,
        "high":   prices + rng.rand(n) * 1.2,
        "low":    prices - rng.rand(n) * 1.2,
        "close":  prices,
        "volume": rng.randint(100_000, 1_000_000, n).astype(float),
    })
    df.set_index("date", inplace=True)
    return df


class TestComputeFactorHistory:
    """compute_factor_history 的列完整性测试"""

    def test_all_expected_columns_present(self):
        """计算后 DataFrame 应包含 FACTOR_META 中定义的全部因子列"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv()
        df = analyzer.compute_factor_history(df)
        for col in FACTOR_META.keys():
            assert col in df.columns, f"缺少因子列: {col}"

    def test_row_count_preserved(self):
        """计算因子后行数不应变化"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(200)
        n_before = len(df)
        df = analyzer.compute_factor_history(df)
        assert len(df) == n_before

    def test_empty_df_returns_empty(self):
        """空 DataFrame 输入时返回空 DataFrame"""
        analyzer = FactorICAnalyzer()
        df = pd.DataFrame()
        result = analyzer.compute_factor_history(df)
        assert result.empty


class TestComputeICRange:
    """IC 值必须在 [-1, 1] 范围内"""

    def test_ic_series_in_valid_range(self):
        """滚动 IC 值必须严格在 [-1, 1] 之间"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(300)
        df = analyzer.compute_factor_history(df)
        fwd = analyzer.compute_forward_returns(df, 1)

        for col in FACTOR_META.keys():
            if col not in df.columns:
                continue
            ic_s = analyzer.compute_ic_series(df[col], fwd, window=60)
            if ic_s.empty:
                continue
            assert ic_s.min() >= -1.0, f"{col} IC 出现 < -1 的值"
            assert ic_s.max() <= 1.0,  f"{col} IC 出现 > +1 的值"


class TestComputeICIR:
    """ICIR 计算测试"""

    def test_icir_positive_from_correlated_data(self):
        """完全正相关的因子与收益，ICIR 应为正值"""
        analyzer = FactorICAnalyzer()
        n = 200
        rng = np.random.RandomState(0)
        vals = pd.Series(rng.randn(n))
        returns = vals + rng.randn(n) * 0.05  # 高相关性
        ic_s = analyzer.compute_ic_series(vals, returns, window=30)
        result = analyzer.compute_icir(ic_s)
        assert result["icir"] > 0, f"高相关因子的 ICIR 应为正数，实际={result['icir']}"

    def test_icir_empty_series(self):
        """空 IC 序列时应返回全零"""
        analyzer = FactorICAnalyzer()
        result = analyzer.compute_icir(pd.Series(dtype=float))
        assert result["ic_mean"] == 0.0
        assert result["icir"] == 0.0

    def test_ic_positive_rate_in_range(self):
        """IC>0 比率必须在 [0, 1] 之间"""
        analyzer = FactorICAnalyzer()
        ic_s = pd.Series(np.random.randn(100))
        result = analyzer.compute_icir(ic_s)
        assert 0.0 <= result["ic_positive_rate"] <= 1.0


class TestICDecay:
    """IC 衰减曲线结构测试"""

    def test_decay_structure(self):
        """返回结构必须包含所有预测期和因子"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(300)
        df = analyzer.compute_factor_history(df)
        factor_cols = [c for c in FACTOR_META.keys() if c in df.columns]
        decay = analyzer.compute_ic_decay(df, factor_cols)

        assert "forward_days" in decay
        assert "factors" in decay
        assert decay["forward_days"] == FORWARD_DAYS

        for col in factor_cols:
            assert col in decay["factors"], f"衰减表缺少因子: {col}"
            assert len(decay["factors"][col]) == len(FORWARD_DAYS), \
                f"{col} 的衰减序列长度与预测期数量不匹配"

    def test_decay_values_in_range(self):
        """IC 衰减曲线中的所有值必须在 [-1, 1]"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(300)
        df = analyzer.compute_factor_history(df)
        factor_cols = [c for c in FACTOR_META.keys() if c in df.columns]
        decay = analyzer.compute_ic_decay(df, factor_cols)

        for col, vals in decay["factors"].items():
            for v in vals:
                assert -1.0 <= v <= 1.0, f"{col} 第{vals.index(v)}期 IC 超出合理范围: {v}"


class TestFactorCorr:
    """因子相关矩阵测试"""

    def test_diagonal_is_one(self):
        """相关矩阵对角线应全为 1.0"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(300)
        df = analyzer.compute_factor_history(df)
        factor_cols = [c for c in FACTOR_META.keys() if c in df.columns]
        result = analyzer.compute_factor_corr(df, factor_cols)

        matrix = result["matrix"]
        labels = result["labels"]
        assert len(labels) > 0, "标签列表不能为空"
        for i, row in enumerate(matrix):
            assert abs(row[i] - 1.0) < 0.01, f"第 {i} 个因子的自相关应为 1.0，实际={row[i]}"

    def test_matrix_symmetry(self):
        """Spearman 相关矩阵应为对称矩阵"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(300)
        df = analyzer.compute_factor_history(df)
        factor_cols = [c for c in FACTOR_META.keys() if c in df.columns]
        result = analyzer.compute_factor_corr(df, factor_cols)

        matrix = result["matrix"]
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(matrix[i][j] - matrix[j][i]) < 0.01, \
                    f"矩阵不对称: [{i}][{j}]={matrix[i][j]} vs [{j}][{i}]={matrix[j][i]}"

    def test_empty_returns_empty(self):
        """数据不足时返回空矩阵"""
        analyzer = FactorICAnalyzer()
        df = _make_ohlcv(5)  # 少于 min_obs
        result = analyzer.compute_factor_corr(df, ["RSI", "MACD_Hist"])
        assert result["matrix"] == [] or len(result["labels"]) <= 1
