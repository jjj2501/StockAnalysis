"""
GPU管理API路由
提供GPU状态查询、配置管理和训练控制功能
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import torch
import json
import os

from backend.core.gpu_utils import get_gpu_manager, device_info
from backend.core.engine import StockEngine
from backend.config import settings

router = APIRouter(prefix="/gpu", tags=["GPU管理"])

# 数据模型
class GPUStatusResponse(BaseModel):
    """GPU状态响应"""
    device: str
    gpu_available: bool
    gpu_info: Dict[str, Any]
    config: Dict[str, Any]
    timestamp: str

class TrainingConfig(BaseModel):
    """训练配置"""
    use_gpu: Optional[bool] = None
    gpu_device_id: Optional[int] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    model_hidden_dim: Optional[int] = None
    model_num_layers: Optional[int] = None
    sequence_length: Optional[int] = None

class TrainingRequest(BaseModel):
    """训练请求"""
    symbol: str
    config: Optional[TrainingConfig] = None

class TrainingStatus(BaseModel):
    """训练状态"""
    symbol: str
    status: str  # "idle", "training", "completed", "failed"
    progress: Optional[float] = None
    message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# 全局训练状态
training_status: Dict[str, TrainingStatus] = {}

@router.get("/status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """获取GPU状态信息"""
    import datetime
    
    info = device_info()
    
    return GPUStatusResponse(
        device=info["device"],
        gpu_available=info["gpu_available"],
        gpu_info=info["gpu_info"],
        config=info["config"],
        timestamp=datetime.datetime.now().isoformat()
    )

@router.get("/devices")
async def get_gpu_devices():
    """获取所有GPU设备信息"""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "devices": []}
    
    devices = []
    for i in range(torch.cuda.device_count()):
        try:
            device_props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": device_props.total_memory / 1024**3,
                "multi_processor_count": device_props.multi_processor_count,
                "major": device_props.major,
                "minor": device_props.minor,
            })
        except Exception as e:
            devices.append({
                "id": i,
                "error": str(e)
            })
    
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": devices
    }

@router.get("/memory")
async def get_gpu_memory():
    """获取GPU内存使用情况"""
    gpu_manager = get_gpu_manager()
    
    if not gpu_manager.is_gpu_available():
        return {"gpu_available": False}
    
    memory_info = gpu_manager.get_memory_info()
    
    # 获取系统内存信息
    import psutil
    system_memory = psutil.virtual_memory()
    
    return {
        "gpu_available": True,
        "gpu_memory": memory_info,
        "system_memory": {
            "total_gb": system_memory.total / 1024**3,
            "available_gb": system_memory.available / 1024**3,
            "percent_used": system_memory.percent,
            "used_gb": system_memory.used / 1024**3,
            "free_gb": system_memory.free / 1024**3,
        }
    }

@router.post("/config/update")
async def update_training_config(config: TrainingConfig):
    """更新训练配置"""
    # 注意：这里只是更新内存中的配置，重启后会恢复默认值
    # 生产环境中应该保存到配置文件或数据库
    
    updated_fields = {}
    
    if config.use_gpu is not None:
        settings.USE_GPU = config.use_gpu
        updated_fields["use_gpu"] = config.use_gpu
    
    if config.gpu_device_id is not None:
        if torch.cuda.is_available() and config.gpu_device_id < torch.cuda.device_count():
            settings.GPU_DEVICE_ID = config.gpu_device_id
            updated_fields["gpu_device_id"] = config.gpu_device_id
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"GPU设备 {config.gpu_device_id} 不可用"
            )
    
    if config.batch_size is not None and config.batch_size > 0:
        settings.BATCH_SIZE = config.batch_size
        updated_fields["batch_size"] = config.batch_size
    
    if config.epochs is not None and config.epochs > 0:
        settings.EPOCHS = config.epochs
        updated_fields["epochs"] = config.epochs
    
    if config.learning_rate is not None and config.learning_rate > 0:
        settings.LEARNING_RATE = config.learning_rate
        updated_fields["learning_rate"] = config.learning_rate
    
    if config.model_hidden_dim is not None and config.model_hidden_dim > 0:
        settings.MODEL_HIDDEN_DIM = config.model_hidden_dim
        updated_fields["model_hidden_dim"] = config.model_hidden_dim
    
    if config.model_num_layers is not None and config.model_num_layers > 0:
        settings.MODEL_NUM_LAYERS = config.model_num_layers
        updated_fields["model_num_layers"] = config.model_num_layers
    
    if config.sequence_length is not None and config.sequence_length > 0:
        settings.SEQUENCE_LENGTH = config.sequence_length
        updated_fields["sequence_length"] = config.sequence_length
    
    # 重新初始化GPU管理器以应用新配置
    global gpu_manager
    gpu_manager = get_gpu_manager()
    
    return {
        "message": "配置已更新",
        "updated_fields": updated_fields,
        "current_config": {
            "use_gpu": settings.USE_GPU,
            "gpu_device_id": settings.GPU_DEVICE_ID,
            "batch_size": settings.BATCH_SIZE,
            "epochs": settings.EPOCHS,
            "learning_rate": settings.LEARNING_RATE,
            "model_hidden_dim": settings.MODEL_HIDDEN_DIM,
            "model_num_layers": settings.MODEL_NUM_LAYERS,
            "sequence_length": settings.SEQUENCE_LENGTH,
        }
    }

@router.get("/config/current")
async def get_current_config():
    """获取当前训练配置"""
    return {
        "use_gpu": settings.USE_GPU,
        "gpu_device_id": settings.GPU_DEVICE_ID,
        "batch_size": settings.BATCH_SIZE,
        "epochs": settings.EPOCHS,
        "learning_rate": settings.LEARNING_RATE,
        "model_hidden_dim": settings.MODEL_HIDDEN_DIM,
        "model_num_layers": settings.MODEL_NUM_LAYERS,
        "sequence_length": settings.SEQUENCE_LENGTH,
    }

def train_model_async(symbol: str, config: Optional[TrainingConfig] = None):
    """异步训练模型"""
    import datetime
    import traceback
    
    training_id = f"{symbol}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # 更新训练状态
        training_status[training_id] = TrainingStatus(
            symbol=symbol,
            status="training",
            progress=0.0,
            message="正在初始化...",
            start_time=datetime.datetime.now().isoformat()
        )
        
        # 应用配置（如果提供）
        engine_config = {}
        if config:
            if config.use_gpu is not None:
                engine_config["use_gpu"] = config.use_gpu
        
        # 创建训练引擎
        engine = StockEngine(**engine_config)
        
        # 定义进度回调
        def progress_callback(progress, message):
            if training_id in training_status:
                training_status[training_id].progress = progress
                training_status[training_id].message = message
        
        # 开始训练
        training_status[training_id].message = "正在获取数据..."
        result = engine.train(symbol, progress_callback=progress_callback)
        
        if result:
            training_status[training_id].status = "completed"
            training_status[training_id].progress = 100.0
            training_status[training_id].message = "训练完成"
            training_status[training_id].result = result
        else:
            training_status[training_id].status = "failed"
            training_status[training_id].message = "训练失败"
        
    except Exception as e:
        training_status[training_id].status = "failed"
        training_status[training_id].message = f"训练错误: {str(e)}"
        print(f"训练错误: {traceback.format_exc()}")
    
    finally:
        training_status[training_id].end_time = datetime.datetime.now().isoformat()

@router.post("/train", response_model=Dict[str, Any])
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """启动模型训练"""
    symbol = request.symbol.upper()
    
    # 检查是否已经在训练
    for training_id, status in training_status.items():
        if status.symbol == symbol and status.status == "training":
            raise HTTPException(
                status_code=400,
                detail=f"股票 {symbol} 的训练已在进行中"
            )
    
    # 在后台启动训练
    background_tasks.add_task(train_model_async, symbol, request.config)
    
    return {
        "message": f"已开始训练股票 {symbol}",
        "symbol": symbol,
        "training_started": True,
        "note": "训练在后台进行，使用 /gpu/training/status 查看进度"
    }

@router.get("/training/status", response_model=List[TrainingStatus])
async def get_training_status(symbol: Optional[str] = None):
    """获取训练状态"""
    if symbol:
        symbol = symbol.upper()
        return [status for training_id, status in training_status.items() 
                if status.symbol == symbol]
    else:
        return list(training_status.values())

@router.get("/training/{training_id}", response_model=TrainingStatus)
async def get_training_by_id(training_id: str):
    """根据ID获取训练状态"""
    if training_id not in training_status:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    return training_status[training_id]

@router.post("/clear-cache")
async def clear_gpu_cache():
    """清除GPU缓存"""
    gpu_manager = get_gpu_manager()
    gpu_manager.clear_cache()
    
    return {"message": "GPU缓存已清除"}

@router.get("/benchmark")
async def run_gpu_benchmark():
    """运行GPU性能基准测试"""
    import time
    import numpy as np
    
    gpu_manager = get_gpu_manager()
    results = {
        "device": str(gpu_manager.device),
        "gpu_available": gpu_manager.is_gpu_available(),
        "tests": {}
    }
    
    # 测试1: 矩阵乘法性能
    try:
        size = 1024
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        for _ in range(10):
            torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        if gpu_manager.is_gpu_available():
            a_gpu = a_cpu.to(gpu_manager.device)
            b_gpu = b_cpu.to(gpu_manager.device)
            
            # 预热
            torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(10):
                torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        else:
            gpu_time = None
            speedup = None
        
        results["tests"]["matrix_multiplication"] = {
            "matrix_size": size,
            "cpu_time_seconds": cpu_time,
            "gpu_time_seconds": gpu_time,
            "speedup": speedup
        }
        
    except Exception as e:
        results["tests"]["matrix_multiplication"] = {"error": str(e)}
    
    # 测试2: 模型推理性能
    try:
        from backend.core.model import HybridModel
        
        # 创建测试模型
        model = HybridModel(
            input_dim=12,
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )
        
        # CPU测试
        model_cpu = model.to("cpu")
        model_cpu.eval()
        
        test_input_cpu = torch.randn(1, 60, 12)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                model_cpu(test_input_cpu)
        cpu_time = time.time() - start
        
        # GPU测试（如果可用）
        if gpu_manager.is_gpu_available():
            model_gpu = model.to(gpu_manager.device)
            model_gpu.eval()
            
            test_input_gpu = test_input_cpu.to(gpu_manager.device)
            
            # 预热
            with torch.no_grad():
                model_gpu(test_input_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                for _ in range(100):
                    model_gpu(test_input_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        else:
            gpu_time = None
            speedup = None
        
        results["tests"]["model_inference"] = {
            "batch_size": 1,
            "sequence_length": 60,
            "iterations": 100,
            "cpu_time_seconds": cpu_time,
            "gpu_time_seconds": gpu_time,
            "speedup": speedup
        }
        
    except Exception as e:
        results["tests"]["model_inference"] = {"error": str(e)}
    
    return results