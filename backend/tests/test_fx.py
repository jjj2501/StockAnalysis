import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from backend.core.fx import FXManager

@pytest.fixture
def fx_manager(tmp_path):
    # 使用临时目录作为缓存，避免污染真实环境
    return FXManager(cache_dir=tmp_path)

def test_fx_same_currency(fx_manager):
    """测试同币种转换，应该返回全1的序列"""
    start_date = "20250101"
    end_date = "20250105"
    
    # 获取 CNY 到 CNY 的折线
    series = fx_manager.get_fx_rate_series("CNY", "CNY", start_date, end_date)
    
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    # 本来日期区间是 1月1日到 1月5日，freq='B'（工作日）
    # 断言里面所有的值都是 1.0
    assert (series == 1.0).all()

@patch('backend.core.fx.yf.download')
def test_fx_different_currency(mock_download, fx_manager):
    """测试不同币种转换，模拟 yfinance 请求过程"""
    start_date = "2025-01-01"
    end_date = "2025-01-02"
    
    # 构造 yfinance 返回的 mock dataframe
    mock_df = pd.DataFrame({
        'Close': [7.10, 7.15]
    }, index=pd.to_datetime(['2025-01-01', '2025-01-02']))
    
    mock_download.return_value = mock_df
    
    series = fx_manager.get_fx_rate_series("USD", "CNY", start_date, end_date)
    
    # 断言调用到了 yfinance API
    mock_download.assert_called_once()
    
    # 验证返回值是否包含了 mock 的汇率
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert series.iloc[0] == 7.10
    
def test_fx_fallback_when_api_fails(fx_manager):
    """测试当 yfinance 获取失败，并且没有缓存时，应该 fallback 返回全 1"""
    # 让 yf.download 抛出异常
    with patch('backend.core.fx.yf.download', side_effect=Exception("API Error")):
        series = fx_manager.get_fx_rate_series("EUR", "CNY", "2025-01-01", "2025-01-05")
        
        # 即使报错也应该返回一个长度是对的序列，且全为 1.0 以防计算崩溃
        assert isinstance(series, pd.Series)
        assert (series == 1.0).all()
