import os
from huggingface_hub import snapshot_download
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置从国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 指定强制下载的本地存储路径
TARGET_DIR = Path(__file__).parent / "backend" / "data" / "models" / "all-MiniLM-L6-v2"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

logger.info(f"Starting to download {MODEL_ID} to {TARGET_DIR}...")
logger.info("Using mirror: https://hf-mirror.com")

try:
    # 强制将模型快照完整克隆至本地目录
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False, # 防止软链在 Windows 下遭遇权限限制
        resume_download=True          # 允许断点续传
    )
    logger.info("✅ Model download completed successfully! Vector DB is now fully offline-ready.")
except Exception as e:
    logger.error(f"❌ Failed to download model: {e}")
