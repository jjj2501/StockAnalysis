import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.core.data import DataFetcher

@pytest.fixture
def data_fetcher(tmp_path):
    """创建带临时缓存目录的 DataFetcher"""
    return DataFetcher(cache_dir=tmp_path)

def test_cache_path_generation(data_fetcher, tmp_path):
    """验证缓存路径生成"""
    symbol = "600519"
    expected_path = tmp_path / f"{symbol}.parquet"
    assert data_fetcher._get_cache_path(symbol) == expected_path

def test_save_and_load_cache(data_fetcher):
    """验证缓存的保存和加载功能"""
    symbol = "000001"
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "close": [10.0, 10.5],
        "open": [9.8, 10.2]
    })
    
    # 保存缓存
    success = data_fetcher._save_cache(symbol, df)
    assert success is True
    assert data_fetcher._get_cache_path(symbol).exists()
    
    # 加载缓存
    loaded_df = data_fetcher._load_cache(symbol)
    assert loaded_df is not None
    assert len(loaded_df) == 2
    assert loaded_df.iloc[0]["close"] == 10.0

def test_get_stock_data_from_cache_only(data_fetcher, mocker):
    """验证当请求范围在缓存内时，不调用远程 API"""
    symbol = "600000"
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
        "close": [10.0, 11.0]
    })
    data_fetcher._save_cache(symbol, df)
    
    # Mock 远程获取方法，确保它没被调用
    mock_remote = mocker.patch.object(data_fetcher, '_fetch_from_remote')
    
    # 请求 20240101 到 20240110 的数据
    result = data_fetcher.get_stock_data(symbol, start_date="20240101", end_date="20240110")
    
    assert not mock_remote.called
    assert len(result) == 2

def test_batch_data_concurrency(data_fetcher, mocker):
    """验证批量数据获取的并发性（使用 Mock）"""
    symbols = ["000001", "000002", "000003"]
    
    # Mock get_stock_data 返回空 DataFrame
    mock_get = mocker.patch.object(data_fetcher, 'get_stock_data', return_value=pd.DataFrame({"a": [1]}))
    
    results = data_fetcher.get_batch_data(symbols, max_workers=3)
    
    assert len(results) == 3
    assert mock_get.call_count == 3
    for s in symbols:
        assert s in results

def test_error_handling_remote_failure(data_fetcher, mocker):
    """验证远程获取失败时的处理"""
    symbol = "invalid_code"
    mocker.patch.object(data_fetcher, '_fetch_from_remote', return_value=pd.DataFrame())
    
    result = data_fetcher.get_stock_data(symbol)
    assert result.empty
