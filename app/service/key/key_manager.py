import asyncio
import random
import time
from itertools import cycle
from typing import Dict, Union, List, Tuple
from collections import defaultdict

from app.config.config import settings
from app.log.logger import get_key_manager_logger
from app.utils.helpers import redact_key_for_logging

logger = get_key_manager_logger()


class KeyManager:
    def __init__(self, api_keys: list, vertex_api_keys: list):
        self.api_keys = api_keys
        self.vertex_api_keys = vertex_api_keys
        self.key_cycle = cycle(api_keys)
        self.vertex_key_cycle = cycle(vertex_api_keys)
        self.key_cycle_lock = asyncio.Lock()
        self.vertex_key_cycle_lock = asyncio.Lock()
        self.failure_count_lock = asyncio.Lock()
        self.vertex_failure_count_lock = asyncio.Lock()
        self.key_usage_lock = asyncio.Lock()
        self.key_failure_counts: Dict[str, int] = {key: 0 for key in api_keys}
        self.vertex_key_failure_counts: Dict[str, int] = {
            key: 0 for key in vertex_api_keys
        }
        # 存储密钥使用时间戳的字典，用于实现冷却期
        # 格式: {key: {model: [timestamp1, timestamp2, ...]}}
        self.key_usage_timestamps: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        # 模型特定的速率限制配置 {model: requests_per_minute}
        self.model_rate_limits = {
            "gemini-2.5-pro": 4  # gemini-2.5-pro 模型每分钟最多4次请求
        }
        self.MAX_FAILURES = settings.MAX_FAILURES
        self.paid_key = settings.PAID_KEY

    async def get_paid_key(self) -> str:
        return self.paid_key

    async def get_next_key(self) -> str:
        """获取下一个API key"""
        async with self.key_cycle_lock:
            return next(self.key_cycle)

    async def get_next_vertex_key(self) -> str:
        """获取下一个 Vertex Express API key"""
        async with self.vertex_key_cycle_lock:
            return next(self.vertex_key_cycle)

    async def is_key_valid(self, key: str) -> bool:
        """检查key是否有效"""
        async with self.failure_count_lock:
            return self.key_failure_counts[key] < self.MAX_FAILURES

    async def is_vertex_key_valid(self, key: str) -> bool:
        """检查 Vertex key 是否有效"""
        async with self.vertex_failure_count_lock:
            return self.vertex_key_failure_counts[key] < self.MAX_FAILURES

    async def reset_failure_counts(self):
        """重置所有key的失败计数"""
        async with self.failure_count_lock:
            for key in self.key_failure_counts:
                self.key_failure_counts[key] = 0

    async def reset_vertex_failure_counts(self):
        """重置所有 Vertex key 的失败计数"""
        async with self.vertex_failure_count_lock:
            for key in self.vertex_key_failure_counts:
                self.vertex_key_failure_counts[key] = 0

    async def reset_key_failure_count(self, key: str) -> bool:
        """重置指定key的失败计数"""
        async with self.failure_count_lock:
            if key in self.key_failure_counts:
                self.key_failure_counts[key] = 0
                logger.info(f"Reset failure count for key: {redact_key_for_logging(key)}")
                return True
            logger.warning(
                f"Attempt to reset failure count for non-existent key: {key}"
            )
            return False

    async def reset_vertex_key_failure_count(self, key: str) -> bool:
        """重置指定 Vertex key 的失败计数"""
        async with self.vertex_failure_count_lock:
            if key in self.vertex_key_failure_counts:
                self.vertex_key_failure_counts[key] = 0
                logger.info(f"Reset failure count for Vertex key: {redact_key_for_logging(key)}")
                return True
            logger.warning(
                f"Attempt to reset failure count for non-existent Vertex key: {key}"
            )
            return False

    async def get_next_working_key(self, model: str = None) -> str:
        """获取下一可用的API key，考虑冷却期限制
        
        Args:
            model: 请求的模型名称，用于检查模型特定的速率限制
            
        Returns:
            可用的API密钥
        """
        initial_key = await self.get_next_key()
        current_key = initial_key
        attempts = 0
        max_attempts = len(self.api_keys) * 2  # 避免无限循环

        while attempts < max_attempts:
            # 检查密钥是否有效（失败计数未超过阈值）
            if await self.is_key_valid(current_key):
                # 如果指定了模型，检查该密钥是否在冷却期内
                if model and model in self.model_rate_limits:
                    if await self.is_key_in_cooldown(current_key, model):
                        logger.info(f"Key {redact_key_for_logging(current_key)} is in cooldown for model {model}, trying next key")
                        current_key = await self.get_next_key()
                        attempts += 1
                        continue
                
                # 密钥有效且不在冷却期内
                return current_key

            current_key = await self.get_next_key()
            attempts += 1
            if current_key == initial_key and attempts > len(self.api_keys):
                # 如果已经循环了一圈还没找到可用密钥，可能所有密钥都在冷却期
                # 此时返回冷却时间最短的密钥
                if model and model in self.model_rate_limits:
                    cooldown_key = await self.get_key_with_shortest_cooldown(model)
                    if cooldown_key:
                        logger.info(f"All keys in cooldown for model {model}, using key with shortest cooldown")
                        return cooldown_key
                return current_key
        
        # 如果尝试次数过多，返回初始密钥
        logger.warning(f"Could not find available key after {max_attempts} attempts, returning initial key")
        return initial_key

    async def get_next_working_vertex_key(self) -> str:
        """获取下一可用的 Vertex Express API key"""
        initial_key = await self.get_next_vertex_key()
        current_key = initial_key

        while True:
            if await self.is_vertex_key_valid(current_key):
                return current_key

            current_key = await self.get_next_vertex_key()
            if current_key == initial_key:
                return current_key

    async def is_key_in_cooldown(self, api_key: str, model: str) -> bool:
        """检查密钥是否在冷却期内
        
        Args:
            api_key: 要检查的API密钥
            model: 模型名称
            
        Returns:
            如果密钥在冷却期内返回True，否则返回False
        """
        if model not in self.model_rate_limits:
            return False
            
        rate_limit = self.model_rate_limits[model]  # 每分钟允许的请求数
        cooldown_window = 60.0  # 冷却窗口为60秒（1分钟）
        
        async with self.key_usage_lock:
            # 获取该密钥对该模型的使用时间戳
            timestamps = self.key_usage_timestamps[api_key][model]
            
            # 清理过期的时间戳（超过冷却窗口的）
            current_time = time.time()
            valid_timestamps = [ts for ts in timestamps if current_time - ts < cooldown_window]
            self.key_usage_timestamps[api_key][model] = valid_timestamps
            
            # 如果在冷却窗口内的请求数量已达到限制，则密钥在冷却期内
            return len(valid_timestamps) >= rate_limit
    
    async def record_key_usage(self, api_key: str, model: str) -> None:
        """记录密钥使用情况
        
        Args:
            api_key: 使用的API密钥
            model: 使用的模型名称
        """
        async with self.key_usage_lock:
            current_time = time.time()
            self.key_usage_timestamps[api_key][model].append(current_time)
            
            # 记录日志
            if model in self.model_rate_limits:
                count = len(self.key_usage_timestamps[api_key][model])
                limit = self.model_rate_limits[model]
                logger.info(f"Key {redact_key_for_logging(api_key)} used for model {model}: {count}/{limit} requests in current window")
    
    async def get_key_with_shortest_cooldown(self, model: str) -> str:
        """获取冷却时间最短的密钥
        
        Args:
            model: 模型名称
            
        Returns:
            冷却时间最短的密钥
        """
        if model not in self.model_rate_limits:
            return await self.get_first_valid_key()
            
        rate_limit = self.model_rate_limits[model]
        cooldown_window = 60.0  # 冷却窗口为60秒
        current_time = time.time()
        
        best_key = None
        min_wait_time = float('inf')
        
        async with self.key_usage_lock:
            for key in self.api_keys:
                # 跳过无效密钥
                if self.key_failure_counts.get(key, 0) >= self.MAX_FAILURES:
                    continue
                    
                timestamps = self.key_usage_timestamps[key][model]
                valid_timestamps = [ts for ts in timestamps if current_time - ts < cooldown_window]
                
                # 如果请求数未达到限制，可以立即使用
                if len(valid_timestamps) < rate_limit:
                    return key
                    
                # 计算最早可用时间
                if valid_timestamps:
                    # 按时间排序
                    valid_timestamps.sort()
                    # 最早的时间戳过期后可以发送新请求
                    wait_time = (valid_timestamps[0] + cooldown_window) - current_time
                    if wait_time < min_wait_time:
                        min_wait_time = wait_time
                        best_key = key
        
        return best_key if best_key else await self.get_first_valid_key()
    
    async def handle_api_failure(self, api_key: str, retries: int, error_type: str = None, model: str = None) -> str:
        """处理API调用失败
        
        Args:
            api_key: 失败的API密钥
            retries: 当前重试次数
            error_type: 错误类型，用于智能处理不同类型的错误
            model: 请求的模型名称，用于检查模型特定的速率限制
            
        Returns:
            新的API密钥，如果没有可用的密钥则返回空字符串
        """
        # 根据错误类型决定增加的失败计数
        failure_increment = 1
        
        # 判断是否是并发请求问题
        is_concurrent_issue = error_type and ("No parts found" in error_type or "parts" in error_type or "No content was returned" in error_type)
        
        # 对于并发限制或资源竞争类错误，使用更小的失败计数增量
        if is_concurrent_issue:
            failure_increment = 0.25  # 对于并发问题，只增加0.25个失败计数
            logger.info(f"Detected concurrent request issue for key {redact_key_for_logging(api_key)}, using reduced failure increment (0.25)")
        elif retries <= 1:  # 首次重试时使用较小的失败计数增量
            failure_increment = 0.5
            logger.info(f"First retry for key {redact_key_for_logging(api_key)}, using reduced failure increment (0.5)")
        else:
            logger.info(f"Multiple retries for key {redact_key_for_logging(api_key)}, using standard failure increment (1.0)")
        
        async with self.failure_count_lock:
            # 使用浮点数存储失败计数，以支持部分增量
            current_count = self.key_failure_counts.get(api_key, 0)
            new_count = current_count + failure_increment
            self.key_failure_counts[api_key] = new_count
            
            if new_count >= self.MAX_FAILURES:
                logger.warning(
                    f"API key {redact_key_for_logging(api_key)} has failed {self.MAX_FAILURES} times"
                )
        
        # 如果是并发问题，可以考虑在一定时间后重试同一个密钥
        if is_concurrent_issue:
            # 对于并发问题，根据重试次数动态调整重用同一密钥的概率
            reuse_probability = 0.6 if retries <= 1 else 0.4
            if random.random() < reuse_probability:
                logger.info(f"Reusing same key {redact_key_for_logging(api_key)} despite concurrent issue (probability: {reuse_probability})")
                return api_key
            else:
                logger.info(f"Switching key despite concurrent issue (probability: {1-reuse_probability})")
        # 对于首次重试的其他错误，也给予一定概率重用同一密钥
        elif retries <= 1 and random.random() < 0.2:
            logger.info(f"Reusing same key {redact_key_for_logging(api_key)} for first retry of non-concurrent error")
            return api_key
            
        if retries < settings.MAX_RETRIES:
            return await self.get_next_working_key(model)
        else:
            return ""

    async def handle_vertex_api_failure(self, api_key: str, retries: int) -> str:
        """处理 Vertex Express API 调用失败"""
        async with self.vertex_failure_count_lock:
            self.vertex_key_failure_counts[api_key] += 1
            if self.vertex_key_failure_counts[api_key] >= self.MAX_FAILURES:
                logger.warning(
                    f"Vertex Express API key {redact_key_for_logging(api_key)} has failed {self.MAX_FAILURES} times"
                )

    def get_fail_count(self, key: str) -> int:
        """获取指定密钥的失败次数"""
        return self.key_failure_counts.get(key, 0)

    def get_vertex_fail_count(self, key: str) -> int:
        """获取指定 Vertex 密钥的失败次数"""
        return self.vertex_key_failure_counts.get(key, 0)

    async def get_all_keys_with_fail_count(self) -> dict:
        """获取所有API key及其失败次数"""
        all_keys = {}
        async with self.failure_count_lock:
            for key in self.api_keys:
                all_keys[key] = self.key_failure_counts.get(key, 0)
        
        valid_keys = {k: v for k, v in all_keys.items() if v < self.MAX_FAILURES}
        invalid_keys = {k: v for k, v in all_keys.items() if v >= self.MAX_FAILURES}
        
        return {"valid_keys": valid_keys, "invalid_keys": invalid_keys, "all_keys": all_keys}

    async def get_keys_by_status(self) -> dict:
        """获取分类后的API key列表，包括失败次数"""
        valid_keys = {}
        invalid_keys = {}

        async with self.failure_count_lock:
            for key in self.api_keys:
                fail_count = self.key_failure_counts[key]
                if fail_count < self.MAX_FAILURES:
                    valid_keys[key] = fail_count
                else:
                    invalid_keys[key] = fail_count

        return {"valid_keys": valid_keys, "invalid_keys": invalid_keys}

    async def get_vertex_keys_by_status(self) -> dict:
        """获取分类后的 Vertex Express API key 列表，包括失败次数"""
        valid_keys = {}
        invalid_keys = {}

        async with self.vertex_failure_count_lock:
            for key in self.vertex_api_keys:
                fail_count = self.vertex_key_failure_counts[key]
                if fail_count < self.MAX_FAILURES:
                    valid_keys[key] = fail_count
                else:
                    invalid_keys[key] = fail_count
        return {"valid_keys": valid_keys, "invalid_keys": invalid_keys}

    async def get_first_valid_key(self) -> str:
        """获取第一个有效的API key"""
        async with self.failure_count_lock:
            for key in self.key_failure_counts:
                if self.key_failure_counts[key] < self.MAX_FAILURES:
                    return key
        if self.api_keys:
            return self.api_keys[0]
        if not self.api_keys:
            logger.warning("API key list is empty, cannot get first valid key.")
            return ""
        return self.api_keys[0]

    async def get_random_valid_key(self) -> str:
        """获取随机的有效API key"""
        valid_keys = []
        async with self.failure_count_lock:
            for key in self.key_failure_counts:
                if self.key_failure_counts[key] < self.MAX_FAILURES:
                    valid_keys.append(key)
        
        if valid_keys:
            return random.choice(valid_keys)
        
        # 如果没有有效的key，返回第一个key作为fallback
        if self.api_keys:
            logger.warning("No valid keys available, returning first key as fallback.")
            return self.api_keys[0]
        
        logger.warning("API key list is empty, cannot get random valid key.")
        return ""


_singleton_instance = None
_singleton_lock = asyncio.Lock()
_preserved_failure_counts: Union[Dict[str, int], None] = None
_preserved_vertex_failure_counts: Union[Dict[str, int], None] = None
_preserved_old_api_keys_for_reset: Union[list, None] = None
_preserved_vertex_old_api_keys_for_reset: Union[list, None] = None
_preserved_next_key_in_cycle: Union[str, None] = None
_preserved_vertex_next_key_in_cycle: Union[str, None] = None


async def get_key_manager_instance(
    api_keys: list = None, vertex_api_keys: list = None
) -> KeyManager:
    """
    获取 KeyManager 单例实例。

    如果尚未创建实例，将使用提供的 api_keys,vertex_api_keys 初始化 KeyManager。
    如果已创建实例，则忽略 api_keys 参数，返回现有单例。
    如果在重置后调用，会尝试恢复之前的状态（失败计数、循环位置）。
    """
    global _singleton_instance, _preserved_failure_counts, _preserved_vertex_failure_counts, _preserved_old_api_keys_for_reset, _preserved_vertex_old_api_keys_for_reset, _preserved_next_key_in_cycle, _preserved_vertex_next_key_in_cycle

    async with _singleton_lock:
        if _singleton_instance is None:
            if api_keys is None:
                raise ValueError(
                    "API keys are required to initialize or re-initialize the KeyManager instance."
                )
            if vertex_api_keys is None:
                raise ValueError(
                    "Vertex Express API keys are required to initialize or re-initialize the KeyManager instance."
                )

            if not api_keys:
                logger.warning(
                    "Initializing KeyManager with an empty list of API keys."
                )
            if not vertex_api_keys:
                logger.warning(
                    "Initializing KeyManager with an empty list of Vertex Express API keys."
                )

            _singleton_instance = KeyManager(api_keys, vertex_api_keys)
            logger.info(
                f"KeyManager instance created/re-created with {len(api_keys)} API keys and {len(vertex_api_keys)} Vertex Express API keys."
            )

            # 1. 恢复失败计数
            if _preserved_failure_counts:
                current_failure_counts = {
                    key: 0 for key in _singleton_instance.api_keys
                }
                for key, count in _preserved_failure_counts.items():
                    if key in current_failure_counts:
                        current_failure_counts[key] = count
                _singleton_instance.key_failure_counts = current_failure_counts
                logger.info("Inherited failure counts for applicable keys.")
            _preserved_failure_counts = None

            if _preserved_vertex_failure_counts:
                current_vertex_failure_counts = {
                    key: 0 for key in _singleton_instance.vertex_api_keys
                }
                for key, count in _preserved_vertex_failure_counts.items():
                    if key in current_vertex_failure_counts:
                        current_vertex_failure_counts[key] = count
                _singleton_instance.vertex_key_failure_counts = (
                    current_vertex_failure_counts
                )
                logger.info("Inherited failure counts for applicable Vertex keys.")
            _preserved_vertex_failure_counts = None

            # 2. 调整 key_cycle 的起始点
            start_key_for_new_cycle = None
            if (
                _preserved_old_api_keys_for_reset
                and _preserved_next_key_in_cycle
                and _singleton_instance.api_keys
            ):
                try:
                    start_idx_in_old = _preserved_old_api_keys_for_reset.index(
                        _preserved_next_key_in_cycle
                    )

                    for i in range(len(_preserved_old_api_keys_for_reset)):
                        current_old_key_idx = (start_idx_in_old + i) % len(
                            _preserved_old_api_keys_for_reset
                        )
                        key_candidate = _preserved_old_api_keys_for_reset[
                            current_old_key_idx
                        ]
                        if key_candidate in _singleton_instance.api_keys:
                            start_key_for_new_cycle = key_candidate
                            break
                except ValueError:
                    logger.warning(
                        f"Preserved next key '{_preserved_next_key_in_cycle}' not found in preserved old API keys. "
                        "New cycle will start from the beginning of the new list."
                    )
                except Exception as e:
                    logger.error(
                        f"Error determining start key for new cycle from preserved state: {e}. "
                        "New cycle will start from the beginning."
                    )

            if start_key_for_new_cycle and _singleton_instance.api_keys:
                try:
                    target_idx = _singleton_instance.api_keys.index(
                        start_key_for_new_cycle
                    )
                    for _ in range(target_idx):
                        next(_singleton_instance.key_cycle)
                    logger.info(
                        f"Key cycle in new instance advanced. Next call to get_next_key() will yield: {start_key_for_new_cycle}"
                    )
                except ValueError:
                    logger.warning(
                        f"Determined start key '{start_key_for_new_cycle}' not found in new API keys during cycle advancement. "
                        "New cycle will start from the beginning."
                    )
                except StopIteration:
                    logger.error(
                        "StopIteration while advancing key cycle, implies empty new API key list previously missed."
                    )
                except Exception as e:
                    logger.error(
                        f"Error advancing new key cycle: {e}. Cycle will start from beginning."
                    )
            else:
                if _singleton_instance.api_keys:
                    logger.info(
                        "New key cycle will start from the beginning of the new API key list (no specific start key determined or needed)."
                    )
                else:
                    logger.info(
                        "New key cycle not applicable as the new API key list is empty."
                    )

            # 清理所有保存的状态
            _preserved_old_api_keys_for_reset = None
            _preserved_next_key_in_cycle = None

            # 3. 调整 vertex_key_cycle 的起始点
            start_key_for_new_vertex_cycle = None
            if (
                _preserved_vertex_old_api_keys_for_reset
                and _preserved_vertex_next_key_in_cycle
                and _singleton_instance.vertex_api_keys
            ):
                try:
                    start_idx_in_old = _preserved_vertex_old_api_keys_for_reset.index(
                        _preserved_vertex_next_key_in_cycle
                    )

                    for i in range(len(_preserved_vertex_old_api_keys_for_reset)):
                        current_old_key_idx = (start_idx_in_old + i) % len(
                            _preserved_vertex_old_api_keys_for_reset
                        )
                        key_candidate = _preserved_vertex_old_api_keys_for_reset[
                            current_old_key_idx
                        ]
                        if key_candidate in _singleton_instance.vertex_api_keys:
                            start_key_for_new_vertex_cycle = key_candidate
                            break
                except ValueError:
                    logger.warning(
                        f"Preserved next key '{_preserved_vertex_next_key_in_cycle}' not found in preserved old Vertex Express API keys. "
                        "New cycle will start from the beginning of the new list."
                    )
                except Exception as e:
                    logger.error(
                        f"Error determining start key for new Vertex key cycle from preserved state: {e}. "
                        "New cycle will start from the beginning."
                    )

            if start_key_for_new_vertex_cycle and _singleton_instance.vertex_api_keys:
                try:
                    target_idx = _singleton_instance.vertex_api_keys.index(
                        start_key_for_new_vertex_cycle
                    )
                    for _ in range(target_idx):
                        next(_singleton_instance.vertex_key_cycle)
                    logger.info(
                        f"Vertex key cycle in new instance advanced. Next call to get_next_vertex_key() will yield: {start_key_for_new_vertex_cycle}"
                    )
                except ValueError:
                    logger.warning(
                        f"Determined start key '{start_key_for_new_vertex_cycle}' not found in new Vertex Express API keys during cycle advancement. "
                        "New cycle will start from the beginning."
                    )
                except StopIteration:
                    logger.error(
                        "StopIteration while advancing Vertex key cycle, implies empty new Vertex Express API key list previously missed."
                    )
                except Exception as e:
                    logger.error(
                        f"Error advancing new Vertex key cycle: {e}. Cycle will start from beginning."
                    )
            else:
                if _singleton_instance.vertex_api_keys:
                    logger.info(
                        "New Vertex key cycle will start from the beginning of the new Vertex Express API key list (no specific start key determined or needed)."
                    )
                else:
                    logger.info(
                        "New Vertex key cycle not applicable as the new Vertex Express API key list is empty."
                    )

            # 清理所有保存的状态
            _preserved_vertex_old_api_keys_for_reset = None
            _preserved_vertex_next_key_in_cycle = None

        return _singleton_instance


async def reset_key_manager_instance():
    """
    重置 KeyManager 单例实例。
    将保存当前实例的状态（失败计数、旧 API keys、下一个 key 提示）
    以供下一次 get_key_manager_instance 调用时恢复。
    """
    global _singleton_instance, _preserved_failure_counts, _preserved_vertex_failure_counts, _preserved_old_api_keys_for_reset, _preserved_vertex_old_api_keys_for_reset, _preserved_next_key_in_cycle, _preserved_vertex_next_key_in_cycle
    async with _singleton_lock:
        if _singleton_instance:
            # 1. 保存失败计数
            _preserved_failure_counts = _singleton_instance.key_failure_counts.copy()
            _preserved_vertex_failure_counts = (
                _singleton_instance.vertex_key_failure_counts.copy()
            )

            # 2. 保存旧的 API keys 列表
            _preserved_old_api_keys_for_reset = _singleton_instance.api_keys.copy()
            _preserved_vertex_old_api_keys_for_reset = (
                _singleton_instance.vertex_api_keys.copy()
            )

            # 3. 保存 key_cycle 的下一个 key 提示
            try:
                if _singleton_instance.api_keys:
                    _preserved_next_key_in_cycle = (
                        await _singleton_instance.get_next_key()
                    )
                else:
                    _preserved_next_key_in_cycle = None
            except StopIteration:
                logger.warning(
                    "Could not preserve next key hint: key cycle was empty or exhausted in old instance."
                )
                _preserved_next_key_in_cycle = None
            except Exception as e:
                logger.error(f"Error preserving next key hint during reset: {e}")
                _preserved_next_key_in_cycle = None

            # 4. 保存 vertex_key_cycle 的下一个 key 提示
            try:
                if _singleton_instance.vertex_api_keys:
                    _preserved_vertex_next_key_in_cycle = (
                        await _singleton_instance.get_next_vertex_key()
                    )
                else:
                    _preserved_vertex_next_key_in_cycle = None
            except StopIteration:
                logger.warning(
                    "Could not preserve next key hint: Vertex key cycle was empty or exhausted in old instance."
                )
                _preserved_vertex_next_key_in_cycle = None
            except Exception as e:
                logger.error(f"Error preserving next key hint during reset: {e}")
                _preserved_vertex_next_key_in_cycle = None

            _singleton_instance = None
            logger.info(
                "KeyManager instance has been reset. State (failure counts, old keys, next key hint) preserved for next instantiation."
            )
        else:
            logger.info(
                "KeyManager instance was not set (or already reset), no reset action performed."
            )
