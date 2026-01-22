from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from backend.core.manager import DataManager

router = APIRouter(prefix="/data", tags=["data"])
manager = DataManager()

@router.get("/stocks")
async def get_monitored_stocks():
    """获取所有受监控股票的同步状态"""
    return manager.get_monitored_stocks()

@router.post("/stocks/{symbol}")
async def add_monitored_stock(symbol: str):
    """添加股票到监控池"""
    success = manager.add_stock(symbol)
    if not success:
        return {"message": f"{symbol} 已经在监控列表中"}
    return {"message": f"成功添加 {symbol}"}

@router.delete("/stocks/{symbol}")
async def remove_monitored_stock(symbol: str):
    """从监控池移除股票"""
    success = manager.remove_stock(symbol)
    if not success:
        raise HTTPException(status_code=404, detail=f"未找到股票 {symbol}")
    return {"message": f"成功移除 {symbol}"}

@router.post("/sync")
async def sync_data(symbol: Optional[str] = None):
    """
    触发同步任务
    :param symbol: 如果提供，只同步单只股票；否则同步全量。
    """
    if symbol:
        success = manager.sync_stock(symbol)
        if not success:
            raise HTTPException(status_code=500, detail=f"同步 {symbol} 失败")
        return {"message": f"{symbol} 同步成功"}
    else:
        results = manager.sync_all()
        return {"message": "批量同步任务已执行", "results": results}
