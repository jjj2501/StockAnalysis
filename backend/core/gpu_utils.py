"""
GPU工具模块
提供GPU检测、管理和优化功能
"""

import torch
import logging
from typing import Optional, Dict, Any
from backend.config import settings

logger = logging.getLogger(__name__)

class GPUManager:
    """GPU管理器"""
    
    def __init__(self, use_gpu: Optional[bool] = None, device_id: Optional[int] = None):
        """
        初始化GPU管理器
        
        Args:
            use_gpu: 是否使用GPU，None则使用配置
            device_id: GPU设备ID，None则使用配置
        """
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.device_id = device_id if device_id is not None else settings.GPU_DEVICE_ID
        self.device = None
        self.gpu_info = {}
        
        self._init_device()
        self._log_gpu_info()
    
    def _init_device(self):
        """初始化设备"""
        if self.use_gpu and torch.cuda.is_available():
            try:
                # 检查指定的GPU设备是否可用
                if self.device_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{self.device_id}")
                    torch.cuda.set_device(self.device_id)
                    logger.info(f"使用GPU设备: cuda:{self.device_id}")
                else:
                    logger.warning(f"GPU设备 {self.device_id} 不可用，使用cuda:0")
                    self.device = torch.device("cuda:0")
                    torch.cuda.set_device(0)
            except Exception as e:
                logger.error(f"GPU初始化失败: {e}，回退到CPU")
                self.device = torch.device("cpu")
                self.use_gpu = False
        else:
            self.device = torch.device("cpu")
            if self.use_gpu and not torch.cuda.is_available():
                logger.warning("CUDA不可用，使用CPU")
            else:
                logger.info("使用CPU设备")
    
    def _log_gpu_info(self):
        """记录GPU信息"""
        if self.use_gpu and torch.cuda.is_available():
            try:
                self.gpu_info = {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(self.device_id),
                    "cuda_version": torch.version.cuda,
                    "memory_allocated": torch.cuda.memory_allocated(self.device_id) / 1024**2,  # MB
                    "memory_reserved": torch.cuda.memory_reserved(self.device_id) / 1024**2,  # MB
                    "memory_total": torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3,  # GB
                }
                
                logger.info(f"GPU设备数量: {self.gpu_info['device_count']}")
                logger.info(f"当前GPU: {self.gpu_info['device_name']}")
                logger.info(f"GPU内存: {self.gpu_info['memory_total']:.2f} GB")
                logger.info(f"已分配内存: {self.gpu_info['memory_allocated']:.2f} MB")
                logger.info(f"保留内存: {self.gpu_info['memory_reserved']:.2f} MB")
                
            except Exception as e:
                logger.error(f"获取GPU信息失败: {e}")
                self.gpu_info = {"error": str(e)}
        else:
            self.gpu_info = {"cuda_available": False}
    
    def get_device(self):
        """获取当前设备"""
        return self.device
    
    def is_gpu_available(self):
        """检查GPU是否可用"""
        return self.use_gpu and torch.cuda.is_available()
    
    def clear_cache(self):
        """清除GPU缓存"""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清除")
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息（MB）"""
        if self.is_gpu_available():
            try:
                return {
                    "allocated": torch.cuda.memory_allocated(self.device_id) / 1024**2,
                    "reserved": torch.cuda.memory_reserved(self.device_id) / 1024**2,
                    "max_allocated": torch.cuda.max_memory_allocated(self.device_id) / 1024**2,
                    "max_reserved": torch.cuda.max_memory_reserved(self.device_id) / 1024**2,
                }
            except Exception as e:
                logger.error(f"获取GPU内存信息失败: {e}")
                return {"error": str(e)}
        return {"cuda_available": False}
    
    def optimize_for_training(self, model, batch_size: Optional[int] = None):
        """
        为训练优化GPU设置
        
        Args:
            model: PyTorch模型
            batch_size: 批次大小，None则使用配置
        """
        if not self.is_gpu_available():
            return model
        
        batch_size = batch_size or settings.BATCH_SIZE
        
        try:
            # 将模型移动到GPU
            model = model.to(self.device)
            
            # 启用cudnn基准测试以优化卷积操作
            torch.backends.cudnn.benchmark = True
            
            # 设置cudnn确定性（如果需要可重复结果）
            # torch.backends.cudnn.deterministic = True
            
            logger.info(f"模型已移动到 {self.device}")
            logger.info(f"批次大小: {batch_size}")
            
            return model
            
        except Exception as e:
            logger.error(f"GPU优化失败: {e}")
            # 回退到CPU
            self.device = torch.device("cpu")
            model = model.to(self.device)
            return model
    
    def check_memory_sufficient(self, model_size_mb: float, batch_size: int, sequence_length: int) -> bool:
        """
        检查GPU内存是否足够
        
        Args:
            model_size_mb: 模型大小（MB）
            batch_size: 批次大小
            sequence_length: 序列长度
            
        Returns:
            bool: 内存是否足够
        """
        if not self.is_gpu_available():
            return True  # CPU总是"足够"
        
        try:
            # 估算所需内存
            # 模型参数内存
            model_memory = model_size_mb
            
            # 激活和梯度内存（粗略估算）
            # 假设每个参数需要存储激活和梯度
            activation_memory = model_size_mb * 2  # 前向和后向
            
            # 批次数据内存
            # 假设输入特征维度为12（如当前配置），float32（4字节）
            data_memory = batch_size * sequence_length * 12 * 4 / 1024**2  # MB
            
            total_estimated = model_memory + activation_memory + data_memory
            
            # 获取可用内存
            memory_info = self.get_memory_info()
            if "error" in memory_info:
                return True  # 如果无法获取内存信息，假设足够
            
            available_memory = self.gpu_info["memory_total"] * 1024 - memory_info["reserved"]  # MB
            
            logger.info(f"估算所需内存: {total_estimated:.2f} MB")
            logger.info(f"可用GPU内存: {available_memory:.2f} MB")
            
            # 保留20%的安全余量
            return total_estimated < available_memory * 0.8
            
        except Exception as e:
            logger.error(f"内存检查失败: {e}")
            return True  # 如果检查失败，假设足够

# 全局GPU管理器实例
gpu_manager = GPUManager()

def get_gpu_manager() -> GPUManager:
    """获取全局GPU管理器"""
    return gpu_manager

def device_info() -> Dict[str, Any]:
    """获取设备信息"""
    return {
        "device": str(gpu_manager.device),
        "gpu_available": gpu_manager.is_gpu_available(),
        "gpu_info": gpu_manager.gpu_info,
        "config": {
            "use_gpu": settings.USE_GPU,
            "gpu_device_id": settings.GPU_DEVICE_ID,
            "batch_size": settings.BATCH_SIZE,
            "epochs": settings.EPOCHS,
            "learning_rate": settings.LEARNING_RATE,
        }
    }