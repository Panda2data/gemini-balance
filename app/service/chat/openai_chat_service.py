# app/services/chat_service.py

import asyncio
import datetime
import json
import re
import time
import uuid
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from app.config.config import settings
from app.core.constants import GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
from app.database.services import (
    add_error_log,
    add_request_log,
)
from app.domain.openai_models import ChatRequest, ImageGenerationRequest
from app.handler.message_converter import OpenAIMessageConverter
from app.handler.response_handler import OpenAIResponseHandler
from app.handler.stream_optimizer import openai_optimizer
from app.log.logger import get_openai_logger
from app.service.client.api_client import GeminiApiClient
from app.service.chat.request_queue import request_queue
from app.service.image.image_create_service import ImageCreateService
from app.service.key.key_manager import KeyManager

logger = get_openai_logger()


def _has_media_parts(messages: List[Dict[str, Any]]) -> bool:
    """判断消息是否包含多媒体部分"""
    for message in messages:
        if "parts" in message:
            for part in message["parts"]:
                if "image_url" in part or "inline_data" in part:
                    return True
    return False


def _clean_json_schema_properties(obj: Any) -> Any:
    """清理JSON Schema中Gemini API不支持的字段"""
    if not isinstance(obj, dict):
        return obj
    
    # Gemini API不支持的JSON Schema字段
    unsupported_fields = {
        "exclusiveMaximum", "exclusiveMinimum", "const", "examples", 
        "contentEncoding", "contentMediaType", "if", "then", "else",
        "allOf", "anyOf", "oneOf", "not", "definitions", "$schema",
        "$id", "$ref", "$comment", "readOnly", "writeOnly"
    }
    
    cleaned = {}
    for key, value in obj.items():
        if key in unsupported_fields:
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_json_schema_properties(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_json_schema_properties(item) for item in value]
        else:
            cleaned[key] = value
    
    return cleaned


def _build_tools(
    request: ChatRequest, messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """构建工具"""
    tool = dict()
    model = request.model

    if (
        settings.TOOLS_CODE_EXECUTION_ENABLED
        and not (
            model.endswith("-search")
            or "-thinking" in model
            or model.endswith("-image")
            or model.endswith("-image-generation")
        )
        and not _has_media_parts(messages)
    ):
        tool["codeExecution"] = {}
        logger.debug("Code execution tool enabled.")
    elif _has_media_parts(messages):
        logger.debug("Code execution tool disabled due to media parts presence.")

    if model.endswith("-search"):
        tool["googleSearch"] = {}
        
    real_model = _get_real_model(model)
    if real_model in settings.URL_CONTEXT_MODELS and settings.URL_CONTEXT_ENABLED:
        tool["urlContext"] = {}

    # 将 request 中的 tools 合并到 tools 中
    if request.tools:
        function_declarations = []
        for item in request.tools:
            if not item or not isinstance(item, dict):
                continue

            if item.get("type", "") == "function" and item.get("function"):
                function = deepcopy(item.get("function"))
                parameters = function.get("parameters", {})
                if parameters.get("type") == "object" and not parameters.get(
                    "properties", {}
                ):
                    function.pop("parameters", None)

                # 清理函数中的不支持字段
                function = _clean_json_schema_properties(function)
                function_declarations.append(function)

        if function_declarations:
            # 按照 function 的 name 去重
            names, functions = set(), []
            for fc in function_declarations:
                if fc.get("name") not in names:
                    if fc.get("name")=="googleSearch":
                        # cherry开启内置搜索时，添加googleSearch工具
                        tool["googleSearch"] = {}
                    else:
                        # 其他函数，添加到functionDeclarations中
                        names.add(fc.get("name"))
                        functions.append(fc)

            tool["functionDeclarations"] = functions

    # 解决 "Tool use with function calling is unsupported" 问题
    if tool.get("functionDeclarations"):
        tool.pop("googleSearch", None)
        tool.pop("codeExecution", None)
        tool.pop("urlContext",None)

    return [tool] if tool else []


def _get_real_model(model: str) -> str:
    if model.endswith("-search"):
        model = model[:-7]
    if model.endswith("-image"):
        model = model[:-6]
    if model.endswith("-non-thinking"):
        model = model[:-13]
    if "-search" in model and "-non-thinking" in model:
        model = model[:-20]
    return model


def _get_safety_settings(model: str) -> List[Dict[str, str]]:
    """获取安全设置"""
    # if (
    #     "2.0" in model
    #     and "gemini-2.0-flash-thinking-exp" not in model
    #     and "gemini-2.0-pro-exp" not in model
    # ):
    if model == "gemini-2.0-flash-exp":
        return GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
    return settings.SAFETY_SETTINGS


def _validate_and_set_max_tokens(
    payload: Dict[str, Any], 
    max_tokens: Optional[int], 
    logger_instance
) -> None:
    """验证并设置 max_tokens 参数"""
    if max_tokens is None:
        return
    
    # 参数验证和处理
    if max_tokens <= 0:
        logger_instance.warning(f"Invalid max_tokens value: {max_tokens}, will not set maxOutputTokens")
        # 不设置 maxOutputTokens，让 Gemini API 使用默认值
    else:
        payload["generationConfig"]["maxOutputTokens"] = max_tokens


def _build_payload(
    request: ChatRequest,
    messages: List[Dict[str, Any]],
    instruction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建请求payload"""
    payload = {
        "contents": messages,
        "generationConfig": {
            "temperature": request.temperature,
            "stopSequences": request.stop,
            "topP": request.top_p,
            "topK": request.top_k,
        },
        "tools": _build_tools(request, messages),
        "safetySettings": _get_safety_settings(request.model),
    }
    
    # 处理 max_tokens 参数
    _validate_and_set_max_tokens(payload, request.max_tokens, logger)
    
    if request.model.endswith("-image") or request.model.endswith("-image-generation"):
        payload["generationConfig"]["responseModalities"] = ["Text", "Image"]
    
    if request.model.endswith("-non-thinking"):
        if "gemini-2.5-pro" in request.model:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 128}
        else:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0} 
    
    elif _get_real_model(request.model) in settings.THINKING_BUDGET_MAP:
        if settings.SHOW_THINKING_PROCESS:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": settings.THINKING_BUDGET_MAP.get(request.model, 1000),
                "includeThoughts": True
            }
        else:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": settings.THINKING_BUDGET_MAP.get(request.model, 1000)}

    if (
        instruction
        and isinstance(instruction, dict)
        and instruction.get("role") == "system"
        and instruction.get("parts")
        and not request.model.endswith("-image")
        and not request.model.endswith("-image-generation")
    ):
        payload["systemInstruction"] = instruction

    return payload


class OpenAIChatService:
    """聊天服务"""

    def __init__(self, base_url: str, key_manager: KeyManager = None):
        self.message_converter = OpenAIMessageConverter()
        self.response_handler = OpenAIResponseHandler(config=None)
        self.api_client = GeminiApiClient(base_url, settings.TIME_OUT)
        self.key_manager = key_manager
        self.image_create_service = ImageCreateService()
        self.request_queue = request_queue

    def _extract_text_from_openai_chunk(self, chunk: Dict[str, Any]) -> str:
        """从OpenAI响应块中提取文本内容"""
        if not chunk.get("choices"):
            return ""

        choice = chunk["choices"][0]
        if "delta" in choice and "content" in choice["delta"]:
            return choice["delta"]["content"]
        return ""

    def _create_char_openai_chunk(
        self, original_chunk: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """创建包含指定文本的OpenAI响应块"""
        chunk_copy = json.loads(json.dumps(original_chunk))
        if chunk_copy.get("choices") and "delta" in chunk_copy["choices"][0]:
            chunk_copy["choices"][0]["delta"]["content"] = text
        return chunk_copy

    async def create_chat_completion(
        self,
        request: ChatRequest,
        api_key: str,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """创建聊天完成"""
        messages, instruction = self.message_converter.convert(request.messages)

        payload = _build_payload(request, messages, instruction)
        
        # 生成请求ID用于日志和跟踪
        request_id = str(uuid.uuid4())
        logger.info(f"Request {request_id}: Processing chat completion for model {request.model}")
        
        try:
            # 将请求添加到队列并等待处理许可
            queue_id, future = await self.request_queue.add_request(request.model, payload, api_key)
            logger.info(f"Request {request_id} queued as {queue_id}")
            
            # 等待队列处理许可
            await asyncio.wait_for(future, timeout=settings.REQUEST_TIMEOUT)
            logger.info(f"Request {request_id} ({queue_id}) received processing permission")
            
            if request.stream:
                return self._handle_stream_completion(request.model, payload, api_key, request_id)
            return await self._handle_normal_completion(request.model, payload, api_key, request_id)
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out waiting in queue")
            raise TimeoutError("Request timed out in queue. The system is currently experiencing high load.")
        except Exception as e:
            logger.error(f"Request {request_id} failed in queue: {str(e)}")
            raise

    async def _handle_normal_completion(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str = None
    ) -> Dict[str, Any]:
        """处理普通聊天完成"""
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None
        response = None
        req_id = request_id or str(uuid.uuid4())  # 确保有请求ID
        
        try:
            # 记录密钥使用情况
            if self.key_manager and model in self.key_manager.model_rate_limits:
                await self.key_manager.record_key_usage(api_key, model)
                logger.info(f"Request {req_id}: Recorded key usage for model {model}")
                
            logger.info(f"Request {req_id}: Sending non-streaming request to model {model}")
            response = await self.api_client.generate_content(payload, model, api_key)
            usage_metadata = response.get("usageMetadata", {})
            is_success = True
            status_code = 200
            logger.info(f"Request {req_id}: Received successful response from model {model}")
            
            # 尝试处理响应，捕获可能的响应处理异常
            try:
                result = self.response_handler.handle_response(
                    response,
                    model,
                    stream=False,
                    finish_reason="stop",
                    usage_metadata=usage_metadata,
                )
                logger.info(f"Request {req_id}: Successfully processed response for model {model}")
                return result
            except Exception as response_error:
                logger.error(f"Request {req_id}: Response processing failed for model {model}: {str(response_error)}")
                
                # 记录详细的错误信息
                if "parts" in str(response_error):
                    logger.error(f"Request {req_id}: Response structure issue - missing or invalid parts")
                    if response.get("candidates"):
                        candidate = response["candidates"][0]
                        content = candidate.get("content", {})
                        logger.error(f"Request {req_id}: Content structure: {content}")
                
                # 检查是否是"No parts found"错误，如果是则尝试重试
                if "parts" in str(response_error) or "No parts found" in str(response_error):
                    logger.warning(f"Request {req_id}: Detected 'No parts found' error, likely due to concurrent requests")
                    # 抛出特定异常以触发重试
                    raise ValueError(f"No parts found in response, triggering retry: {str(response_error)}")
                else:
                    # 重新抛出其他类型的异常
                    raise response_error
                
        except ValueError as ve:
            # 处理特定的"No parts found"错误
            if "No parts found" in str(ve) or "parts" in str(ve):
                is_success = False
                error_log_msg = str(ve)
                logger.error(f"Request {req_id}: API call failed with parts error for model {model}: {error_log_msg}")
                # 如果有KeyManager，尝试获取新的API密钥
                if self.key_manager:
                    try:
                        # 传递错误类型信息给handle_api_failure方法
                        new_key = await self.key_manager.handle_api_failure(api_key, 1, str(ve))
                        if new_key and new_key != api_key:
                            logger.info(f"Request {req_id}: Retrying with new API key due to 'No parts found' error")
                            # 使用新密钥重试
                            return await self._handle_normal_completion(model, payload, new_key, req_id)
                    except Exception as key_error:
                        logger.error(f"Request {req_id}: Error while getting new API key: {str(key_error)}")
                # 如果无法获取新密钥或重试失败，重新抛出异常
                raise ve
            else:
                # 其他ValueError类型，按常规异常处理
                raise
                
        except Exception as e:
            is_success = False
            error_log_msg = str(e)
            logger.error(f"Request {req_id}: API call failed for model {model}: {error_log_msg}")
            
            # 特别记录 max_tokens 相关的错误
            gen_config = payload.get('generationConfig', {})
            if "maxOutputTokens" in gen_config:
                logger.error(f"Request {req_id}: Request had maxOutputTokens: {gen_config['maxOutputTokens']}")
            
            # 如果是响应处理错误，记录更多信息
            if "parts" in error_log_msg:
                logger.error(f"Request {req_id}: This is likely a response processing error")
            
            match = re.search(r"status code (\d+)", error_log_msg)
            status_code = int(match.group(1)) if match else 500

            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="openai-chat-non-stream",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg=payload,
            )
            raise e
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.info(f"Normal completion finished - Success: {is_success}, Latency: {latency_ms}ms")
            
            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )

    async def _fake_stream_logic_impl(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str = None
    ) -> AsyncGenerator[str, None]:
        """处理伪流式 (fake stream) 的核心逻辑"""
        req_id = request_id or str(uuid.uuid4())  # 确保有请求ID
        logger.info(
            f"Request {req_id}: Fake streaming enabled for model: {model}. Calling non-streaming endpoint."
        )
        keep_sending_empty_data = True
        # 重置内容标记
        self._stream_had_content = False
        
        # 记录密钥使用情况
        if self.key_manager and model in self.key_manager.model_rate_limits:
            await self.key_manager.record_key_usage(api_key, model)
            logger.info(f"Request {req_id}: Recorded key usage for model {model}")

        async def send_empty_data_locally() -> AsyncGenerator[str, None]:
            """定期发送空数据以保持连接"""
            while keep_sending_empty_data:
                await asyncio.sleep(settings.FAKE_STREAM_EMPTY_DATA_INTERVAL_SECONDS)
                if keep_sending_empty_data:
                    empty_chunk = self.response_handler.handle_response({}, model, stream=True, finish_reason='stop', usage_metadata=None)
                    yield f"data: {json.dumps(empty_chunk)}\n\n"
                    logger.debug("Sent empty data chunk for fake stream heartbeat.")

        empty_data_generator = send_empty_data_locally()
        api_response_task = asyncio.create_task(
            self.api_client.generate_content(payload, model, api_key)
        )

        try:
            while not api_response_task.done():
                try:
                    next_empty_chunk = await asyncio.wait_for(
                        empty_data_generator.__anext__(), timeout=0.1
                    )
                    yield next_empty_chunk
                except asyncio.TimeoutError:
                    pass
                except (
                    StopAsyncIteration
                ):
                    break

            response = await api_response_task
        finally:
            keep_sending_empty_data = False

        if response and response.get("candidates"):
            try:
                response_data = self.response_handler.handle_response(response, model, stream=True, finish_reason='stop', usage_metadata=response.get("usageMetadata", {}))
                # 标记已处理有效内容
                self._stream_had_content = True
                yield f"data: {json.dumps(response_data)}\n\n"
                logger.info(f"Request {req_id}: Sent full response content for fake stream: {model}")
            except ValueError as ve:
                # 捕获并重新抛出 "No parts found" 错误，让上层处理
                if "No parts found" in str(ve):
                    logger.warning(f"Request {req_id}: {str(ve)}")
                    raise ValueError(f"Request {req_id}: {str(ve)}")
                # 其他 ValueError 也重新抛出
                raise
        else:
            error_message = "Failed to get response from model"
            if (
                response and isinstance(response, dict) and response.get("error")
            ):
                error_details = response.get("error")
                if isinstance(error_details, dict):
                    error_message = error_details.get("message", error_message)

            logger.error(
                f"No candidates or error in response for fake stream model {model}: {response}"
            )
            error_chunk = self.response_handler.handle_response({}, model, stream=True, finish_reason='stop', usage_metadata=None)
            yield f"data: {json.dumps(error_chunk)}\n\n"

    async def _real_stream_logic_impl(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str = None
    ) -> AsyncGenerator[str, None]:
        """处理真实流式 (real stream) 的核心逻辑"""
        req_id = request_id or str(uuid.uuid4())  # 确保有请求ID
        logger.info(f"Request {req_id}: Starting real stream for model: {model}")
        tool_call_flag = False
        usage_metadata = None
        # 重置内容标记
        self._stream_had_content = False
        
        # 记录密钥使用情况
        if self.key_manager and model in self.key_manager.model_rate_limits:
            await self.key_manager.record_key_usage(api_key, model)
            logger.info(f"Request {req_id}: Recorded key usage for model {model}")
            
        async for line in self.api_client.stream_generate_content(
            payload, model, api_key
        ):
            if line.startswith("data:"):
                chunk_str = line[6:]
                if not chunk_str or chunk_str.isspace():
                    logger.debug(
                        f"Received empty data line for model {model}, skipping."
                    )
                    continue
                try:
                    chunk = json.loads(chunk_str)
                    usage_metadata = chunk.get("usageMetadata", {})
                except json.JSONDecodeError:
                    logger.error(
                        f"Request {req_id}: Failed to decode JSON from stream for model {model}: {chunk_str}"
                    )
                    continue
                
                try:
                    openai_chunk = self.response_handler.handle_response(
                        chunk, model, stream=True, finish_reason=None, usage_metadata=usage_metadata
                    )
                except ValueError as ve:
                    # 捕获并重新抛出 "No parts found" 错误，让上层处理
                    if "No parts found" in str(ve):
                        logger.warning(f"Request {req_id}: {str(ve)}")
                        raise ValueError(f"Request {req_id}: {str(ve)}")
                    # 其他 ValueError 也重新抛出
                    raise
                if openai_chunk:
                    text = self._extract_text_from_openai_chunk(openai_chunk)
                    if text and settings.STREAM_OPTIMIZER_ENABLED:
                        async for (
                            optimized_chunk_data
                        ) in openai_optimizer.optimize_stream_output(
                            text,
                            lambda t: self._create_char_openai_chunk(openai_chunk, t),
                            lambda c: f"data: {json.dumps(c)}\n\n",
                        ):
                            # 标记已处理有效内容
                            self._stream_had_content = True
                            yield optimized_chunk_data
                    else:
                        if openai_chunk.get("choices") and openai_chunk["choices"][0].get("delta", {}).get("tool_calls"):
                            tool_call_flag = True

                        # 标记已处理有效内容
                        self._stream_had_content = True
                        yield f"data: {json.dumps(openai_chunk)}\n\n"

        if tool_call_flag:
            yield f"data: {json.dumps(self.response_handler.handle_response({}, model, stream=True, finish_reason='tool_calls', usage_metadata=usage_metadata))}\n\n"
        else:
            yield f"data: {json.dumps(self.response_handler.handle_response({}, model, stream=True, finish_reason='stop', usage_metadata=usage_metadata))}\n\n"

    async def _handle_stream_completion(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str = None
    ) -> AsyncGenerator[str, None]:
        """处理流式聊天完成，添加重试逻辑和假流式支持"""
        retries = 0
        max_retries = settings.MAX_RETRIES
        is_success = False
        status_code = None
        final_api_key = api_key
        req_id = request_id or str(uuid.uuid4())  # 确保有请求ID

        while retries < max_retries:
            start_time = time.perf_counter()
            request_datetime = datetime.datetime.now()
            current_attempt_key = final_api_key

            try:
                stream_generator = None
                if settings.FAKE_STREAM_ENABLED:
                    logger.info(
                        f"Request {req_id}: Using fake stream logic for model: {model}, Attempt: {retries + 1}"
                    )
                    stream_generator = self._fake_stream_logic_impl(
                        model, payload, current_attempt_key, req_id
                    )
                else:
                    logger.info(
                        f"Request {req_id}: Using real stream logic for model: {model}, Attempt: {retries + 1}"
                    )
                    stream_generator = self._real_stream_logic_impl(
                        model, payload, current_attempt_key, req_id
                    )

                async for chunk_data in stream_generator:
                    yield chunk_data

                # 检查是否有实际内容返回
                if not hasattr(self, '_stream_had_content') or self._stream_had_content is not True:
                    logger.warning(f"Request {req_id}: No content was returned in the stream response")
                    # 抛出异常以触发重试
                    raise ValueError("No content was returned in the stream response")
                
                yield "data: [DONE]\n\n"
                logger.info(
                    f"Request {req_id}: Streaming completed successfully for model: {model}, FakeStream: {settings.FAKE_STREAM_ENABLED}, Attempt: {retries + 1}"
                )
                is_success = True
                status_code = 200
                break

            except Exception as e:
                retries += 1
                is_success = False
                error_log_msg = str(e)
                logger.warning(
                    f"Request {req_id}: Streaming API call failed with error: {error_log_msg}. Attempt {retries} of {max_retries} with key {current_attempt_key}"
                )

                match = re.search(r"status code (\d+)", error_log_msg)
                if match:
                    status_code = int(match.group(1))
                else:
                    if isinstance(e, asyncio.TimeoutError):
                        status_code = 408
                    else:
                        status_code = 500
                
                # 检查是否是"No parts found in stream response"错误
                if "No parts found" in error_log_msg or "parts" in error_log_msg or "No content was returned" in error_log_msg:
                    logger.warning(f"Request {req_id}: Detected 'No parts found' or 'No content' error, likely due to concurrent requests")
                    # 增加额外延迟，避免立即重试导致的资源竞争
                    await asyncio.sleep(1 + retries * 1.0)  # 更长的基础延迟和更大的增量

                await add_error_log(
                    gemini_key=current_attempt_key,
                    model_name=model,
                    error_type="openai-chat-stream",
                    error_log=error_log_msg,
                    error_code=status_code,
                    request_msg=payload,
                )

                if self.key_manager:
                    new_api_key = await self.key_manager.handle_api_failure(
                        current_attempt_key, retries, error_log_msg, model
                    )
                    if new_api_key and new_api_key != current_attempt_key:
                        final_api_key = new_api_key
                        logger.info(
                            f"Request {req_id}: Switched to new API key for next attempt: {final_api_key}"
                        )
                    elif not new_api_key:
                        logger.error(
                            f"Request {req_id}: No valid API key available after {retries} retries, ceasing attempts for this request."
                        )
                        break
                else:
                    logger.error(
                        f"Request {req_id}: KeyManager not available, cannot switch API key. Ceasing attempts for this request."
                    )
                    break

                if retries >= max_retries:
                    logger.error(
                        f"Request {req_id}: Max retries ({max_retries}) reached for streaming model {model}."
                    )
            finally:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                await add_request_log(
                    model_name=model,
                    api_key=current_attempt_key,
                    is_success=is_success,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_time=request_datetime,
                )

        if not is_success:
            logger.error(
                f"Streaming failed permanently for model {model} after {retries} attempts."
            )
            yield f"data: {json.dumps({'error': f'Streaming failed after {retries} retries.'})}\n\n"
            yield "data: [DONE]\n\n"

    async def create_image_chat_completion(
        self, request: ChatRequest, api_key: str
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:

        image_generate_request = ImageGenerationRequest()
        image_generate_request.prompt = request.messages[-1]["content"]
        image_res = self.image_create_service.generate_images_chat(
            image_generate_request
        )

        if request.stream:
            return self._handle_stream_image_completion(
                request.model, image_res, api_key
            )
        else:
            return await self._handle_normal_image_completion(
                request.model, image_res, api_key
            )

    async def _handle_stream_image_completion(
        self, model: str, image_data: str, api_key: str
    ) -> AsyncGenerator[str, None]:
        logger.info(f"Starting stream image completion for model: {model}")
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None

        try:
            if image_data:
                openai_chunk = self.response_handler.handle_image_chat_response(
                    image_data, model, stream=True, finish_reason=None
                )
                if openai_chunk:
                    # 提取文本内容
                    text = self._extract_text_from_openai_chunk(openai_chunk)
                    if text:
                        # 使用流式输出优化器处理文本输出
                        async for (
                            optimized_chunk
                        ) in openai_optimizer.optimize_stream_output(
                            text,
                            lambda t: self._create_char_openai_chunk(openai_chunk, t),
                            lambda c: f"data: {json.dumps(c)}\n\n",
                        ):
                            yield optimized_chunk
                    else:
                        # 如果没有文本内容（如图片URL等），整块输出
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
            yield f"data: {json.dumps(self.response_handler.handle_response({}, model, stream=True, finish_reason='stop'))}\n\n"
            logger.info(
                f"Stream image completion finished successfully for model: {model}"
            )
            is_success = True
            status_code = 200
            yield "data: [DONE]\n\n"
        except Exception as e:
            is_success = False
            error_log_msg = f"Stream image completion failed for model {model}: {e}"
            logger.error(error_log_msg)
            status_code = 500
            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="openai-image-stream",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg={"image_data_truncated": image_data[:1000]},
            )
            yield f"data: {json.dumps({'error': error_log_msg})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.info(
                f"Stream image completion for model {model} took {latency_ms} ms. Success: {is_success}"
            )
            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )

    async def _handle_normal_image_completion(
        self, model: str, image_data: str, api_key: str
    ) -> Dict[str, Any]:
        logger.info(f"Starting normal image completion for model: {model}")
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None
        result = None

        try:
            result = self.response_handler.handle_image_chat_response(
                image_data, model, stream=False, finish_reason="stop"
            )
            logger.info(
                f"Normal image completion finished successfully for model: {model}"
            )
            is_success = True
            status_code = 200
            return result
        except Exception as e:
            is_success = False
            error_log_msg = f"Normal image completion failed for model {model}: {e}"
            logger.error(error_log_msg)
            status_code = 500
            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="openai-image-non-stream",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg={"image_data_truncated": image_data[:1000]},
            )
            raise e
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.info(
                f"Normal image completion for model {model} took {latency_ms} ms. Success: {is_success}"
            )
            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )
