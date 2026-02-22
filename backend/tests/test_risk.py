import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from backend.core.risk import RiskManager

@pytest.fixture
def risk_manager():
    return RiskManager()

@patch('backend.core.risk.DataFetcher')
@patch('backend.core.risk.FXManager')
def test_risk_manager_cny_conversion(MockFX, MockData, risk_manager):
    # 配置 mock 的获取器
    mock_fx_instance = MockFX.return_value
    mock_data_instance = MockData.return_value
    
    # 模拟汇率，默认是美元兑人民币为 7.0
    mock_fx_series = pd.Series([7.0] * 252, index=pd.date_range('2024-01-01', periods=252, freq='B'))
    mock_fx_instance.get_fx_rate_series.return_value = mock_fx_series
    
    # 模拟苹果公司股价，默认每天都涨一点
    mock_stock_df = pd.DataFrame({
        'close': [100.0 + i * 0.1 for i in range(252)]
    }, index=pd.date_range('2024-01-01', periods=252, freq='B'))
    mock_stock_df = mock_stock_df.reset_index(names=['date'])
    mock_data_instance.get_stock_data.return_value = mock_stock_df
    
    risk_manager.fx_manager = mock_fx_instance
    risk_manager.fetcher = mock_data_instance

    portfolio = [
        {
            "symbol": "AAPL",
            "shares": 10,
            "price": 100.0,
            "market": "US",
            "currency": "USD"
        }
    ]
    
    result = risk_manager.calculate_portfolio_risk(portfolio, days_history=100)
    
    # 断言成功状态
    assert result.get("status") == "success"
    
    # 检查穿透后的人民币市值 (10股 * $100 * 7.0汇率)
    assert result.get("total_value") == 7000.0
    
    # 检查分解数据
    breakdown = result.get("assets_breakdown")
    assert breakdown is not None
    assert len(breakdown) == 1
    assert breakdown[0]["symbol"] == "AAPL"
    assert breakdown[0]["val_cny"] == 7000.0
    assert breakdown[0]["weight"] == 1.0

def test_risk_manager_empty_portfolio(risk_manager):
    # 空组合应该返回 error
    result = risk_manager.calculate_portfolio_risk([])
    assert "error" in result
