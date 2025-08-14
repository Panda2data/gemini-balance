# app/service/chat/request_queue.py

import asyncio
import time
from typing import Any, Dict, Optional, Tuple

from app.config.config import settings
from app.log.logger import get_openai_logger

logger = get_openai_logger()


class RequestQueueItem:
    """请求队列项"""

    def __init__(self, model: str, payload: Dict[str, Any], api_key: str, request_id: str):
        self.model = model
        self.payload = payload
        self.api_key = api_key
        self.request_id = request_id
        self.timestamp = time.time()
        self.future = asyncio.Future()


class RequestQueue:
    """请求队列管理器"""

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=settings.REQUEST_QUEUE_SIZE)
        self.active_requests = {}
        self.semaphore = asyncio.Semaphore(settings.CONCURRENT_REQUEST_LIMIT)
        self.lock = asyncio.Lock()
        self.request_counter = 0
        self._start_queue_processor()

    def _generate_request_id(self) -> str:
        """生成唯一的请求ID"""
        self.request_counter += 1
        return f"req_{int(time.time())}_{self.request_counter}"

    def _start_queue_processor(self):
        """启动队列处理器"""
        asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """处理队列中的请求"""
        while True:
            try:
                item = await self.queue.get()
                asyncio.create_task(self._handle_request(item))
            except Exception as e:
                logger.error(f"Error processing request from queue: {str(e)}")
                await asyncio.sleep(1)  # 避免在错误情况下过度消耗CPU

    async def _handle_request(self, item: RequestQueueItem):
        """处理单个请求"""
        try:
            async with self.semaphore:
                # 检查请求是否已超时
                if time.time() - item.timestamp > settings.REQUEST_TIMEOUT:
                    if not item.future.done():
                        item.future.set_exception(
                            TimeoutError(f"Request {item.request_id} timed out in queue")
                        )
                    return

                # 将请求标记为活动状态
                async with self.lock:
                    self.active_requests[item.request_id] = item

                # 这里不实际处理请求，只是通知调用者可以开始处理
                if not item.future.done():
                    item.future.set_result(True)

        except Exception as e:
            logger.error(f"Error handling request {item.request_id}: {str(e)}")
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            # 请求完成后从活动请求中移除
            async with self.lock:
                self.active_requests.pop(item.request_id, None)
            self.queue.task_done()

    async def add_request(
        self, model: str, payload: Dict[str, Any], api_key: str
    ) -> Tuple[str, asyncio.Future]:
        """添加请求到队列"""
        request_id = self._generate_request_id()
        item = RequestQueueItem(model, payload, api_key, request_id)

        try:
            # 尝试将请求添加到队列
            await asyncio.wait_for(
                self.queue.put(item), timeout=settings.REQUEST_TIMEOUT
            )
            logger.info(
                f"Request {request_id} for model {model} added to queue. Queue size: {self.queue.qsize()}/{settings.REQUEST_QUEUE_SIZE}"
            )
            return request_id, item.future
        except asyncio.TimeoutError:
            logger.error(f"Timeout adding request {request_id} to queue")
            raise TimeoutError(f"Request queue is full. Please try again later.")
        except Exception as e:
            logger.error(f"Error adding request {request_id} to queue: {str(e)}")
            raise

    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求状态"""
        async with self.lock:
            if request_id in self.active_requests:
                item = self.active_requests[request_id]
                wait_time = time.time() - item.timestamp
                return {
                    "request_id": request_id,
                    "model": item.model,
                    "status": "processing",
                    "wait_time": wait_time,
                }

        # 检查队列中的请求
        for i in range(self.queue.qsize()):
            try:
                item = self.queue._queue[i]  # 直接访问队列内部，不是标准做法但有效
                if item.request_id == request_id:
                    wait_time = time.time() - item.timestamp
                    position = i + 1
                    return {
                        "request_id": request_id,
                        "model": item.model,
                        "status": "queued",
                        "position": position,
                        "wait_time": wait_time,
                    }
            except (IndexError, AttributeError):
                pass

        return None


# 创建全局请求队列实例
request_queue = RequestQueue()