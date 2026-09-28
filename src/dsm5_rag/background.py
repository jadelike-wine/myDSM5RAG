"""把 CPU 密集的同步调用挪出事件循环。

FastAPI/MCP 都跑在同一个事件循环上：embedding 批处理、BM25 建索引与打分、文档
解析都是同步阻塞调用，直接写在 async 路径里会让整个服务在单请求期间无法响应
其他请求。这里统一提供一个线程池和 `run_in_executor` 包装。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# 线程池用于执行同步操作
executor = ThreadPoolExecutor(max_workers=4)


async def run_in_executor(func: Callable[..., Any], *args: Any) -> Any:
    """在当前事件循环的默认线程池之外执行同步函数。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)
