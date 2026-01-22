import asyncio
import json
import logging
from typing import Dict, AsyncGenerator

logger = logging.getLogger(__name__)

class ProgressManager:
    """
    轻量级进度管理中心，支持多任务进度追踪与 SSE 推送
    """
    def __init__(self):
        # task_id -> Queue[str]
        self.queues: Dict[str, asyncio.Queue] = {}

    async def subscribe(self, task_id: str) -> AsyncGenerator[str, None]:
        """
        订阅某个任务的进度流
        """
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()
        
        queue = self.queues[task_id]
        try:
            while True:
                msg = await queue.get()
                yield f"data: {msg}\n\n"
                # 任务完成或报错后结束流
                if '"progress": 100' in msg or '"status": "error"' in msg:
                    break
        finally:
            # 清理队列
            if task_id in self.queues:
                del self.queues[task_id]

    async def update(self, task_id: str, progress: int, status: str, detail: str = ""):
        """
        更新进度消息
        """
        if task_id in self.queues:
            msg = json.dumps({
                "progress": progress,
                "status": status,
                "detail": detail
            })
            await self.queues[task_id].put(msg)
            logger.info(f"Task {task_id} progress updated: {progress}% - {status}")

# 全局单例
progress_manager = ProgressManager()
