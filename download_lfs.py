import urllib.request
import urllib.error
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 大文件存放的绝对路径
MODEL_DIR = Path(__file__).parent / "backend" / "data" / "models" / "all-MiniLM-L6-v2"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"

# 镜像源大文件直链
URL = "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"

def download_model_file():
    logger.info(f"Downloading model.safetensors from {URL} ...")
    try:
        req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(SAFETENSORS_PATH, 'wb') as out_file:
            # 读取并写入
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            downloaded = 0
            block_size = 8192
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # 简单进度
                if downloaded % (1024 * 1024 * 10) == 0:  # 每 10MB 打印一次
                    logger.info(f"Downloaded {downloaded / (1024*1024):.2f} MB...")
                    
        logger.info("✅ SAFETENSORS file downloaded successfully!")
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")

if __name__ == "__main__":
    download_model_file()
