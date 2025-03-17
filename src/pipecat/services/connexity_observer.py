import uuid
import time
import copy
from datetime import datetime
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    UserStoppedSpeakingFrame,
    MetricsFrame,
    TextFrame,
    CancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMMessagesFrame
)
from pipecat.metrics.metrics import TTFBMetricsData, TTSUsageMetricsData
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from loguru import logger
import aiohttp
import json


class PipelineTelemetryObserver(BaseObserver):
    def __init__(self, base_url=None, api_key=None, agent_id='default-agent', sid=None, stream=False):
        self.events = {
            "conversation_start": {
                "timestamp": int(time.time() * 1_000_000_000),
                "iso_time": datetime.now().isoformat()
            }
        }
        self.metrics = {
            "BOT_LLM_TIME": 0.0,
            "BOT_TTS_TIME": 0.0,
            "BOT_STT_TIME": 0.0,
            "BOT_TTS_CHARACTERS": 0,
            "FUNCTION_CALLS": []
        }
        # API and configuration settings
        self.api_key = api_key
        self.base_url = base_url
        self.agent_id = agent_id
        self.stream = stream
        self.sid = sid
        
        # Message tracking
        self.messages = []
        self.message_timestamps = {}
        
        # State tracking
        self.end_update_sent = False
        self.start_update_sent = False
        self.collecting_assistant_message = False
        self.current_assistant_message = ""
        self._current_function = None
        
        # User message timing handling
        self.last_user_message_timestamp = 0
        
        # Service tracking
        self.llm_model = None
        self.llm_provider = None   
        self.transcriber = None
        self.voice_provider = None
        
        # Deduplication
        self.processed_function_results = set()
        self.processed_bot_stopped_frames = set()
        self.processed_user_stopped_frames = set()
        self.sent_message_contents = set()

        # Context tracking
        self.context_messages = []

    async def _send_message_update(self, event_type: str, data: dict):
        """Send message updates to the local server with all relevant messages"""
        try:
            if event_type == "conversation_update" and data.get("message") and data["message"].get("role") == "user":
                content = data["message"].get("content", "")
                if content in self.sent_message_contents:
                    return
                self.sent_message_contents.add(content)
                
            url = f"{self.base_url}/process/sdk"

            messages = []

            if event_type == "conversation_update" and not self.stream:
                if not self.start_update_sent:
                    self.start_update_sent = True
                else:
                    return
            
            user_messages = self._get_user_messages_from_context()
            assistant_messages = self._get_assistant_messages_from_context()

            if event_type == "conversation_update" and data.get("message"):
                msg = data["message"]
                formatted_msg = {
                    "id": str(uuid.uuid4()),
                    "role": msg.get("role", ""),
                    "seconds_from_start": self._calculate_seconds_from_start(msg)
                }

                if msg.get("role") == "user":
                    user_ctx_messages = user_messages
                    latest_ctx_msg = user_ctx_messages[-1] if user_ctx_messages else None
                    
                    if latest_ctx_msg and "content" in latest_ctx_msg:
                        formatted_msg["content"] = latest_ctx_msg.get("content", "")
                    elif "content" in msg:
                        formatted_msg["content"] = msg.get("content", "")
                elif "content" in msg:
                    formatted_msg["content"] = msg.get("content", "")
                
                if msg.get("role") == "assistant":
                    assistant_ctx_messages = assistant_messages
                    latest_ctx_msg = assistant_ctx_messages[-1] if assistant_ctx_messages else None
                    
                    if latest_ctx_msg and "content" in latest_ctx_msg:
                        formatted_msg["content"] = latest_ctx_msg.get("content", "")
                    elif "content" in msg:
                        formatted_msg["content"] = msg.get("content", "")
                        
                    formatted_msg["time_to_first_audio"] = self._calculate_time_to_first_audio()
                    formatted_msg["latency"] = {
                        "tts": int(self.metrics.get("BOT_TTS_TIME", 0) * 1000),
                        "llm": int(self.metrics.get("BOT_LLM_TIME", 0) * 1000),
                        "stt": int(self.metrics.get("BOT_STT_TIME", 0) * 1000)
                    }

                    if "tool_calls" in msg:
                        formatted_msg["tool_calls"] = msg["tool_calls"]

                elif msg.get("role") == "tool_call":
                    formatted_msg["tool_call_id"] = msg.get("tool_call_id", "")
                    formatted_msg["name"] = msg.get("name", "")
                    formatted_msg["arguments"] = msg.get("arguments", {})
                    formatted_msg["function"] = {
                        "name": msg.get("name", ""),
                        "arguments": json.dumps(msg.get("arguments", {}))
                    }

                elif msg.get("role") == "tool" or msg.get("role") == "tool_result":
                    formatted_msg["role"] = "tool"
                    result = msg.get("result")
                    if result is not None:
                        formatted_msg["content"] = json.dumps(result)
                    else:
                        formatted_msg["content"] = msg.get("content", "")
                    formatted_msg["tool_call_id"] = msg.get("tool_call_id", "")

                messages.append(formatted_msg)
            
            if event_type == "end_of_call" and not self.stream:
                for msg in self.messages:
                    formatted_msg = {
                        "id": str(uuid.uuid4()),
                        "role": msg.get("role", ""),
                        "seconds_from_start": self._calculate_seconds_from_start(msg)
                    }

                    if msg.get("role") == "user":
                        user_ctx_messages = user_messages
                        latest_ctx_msg = user_ctx_messages[-1] if user_ctx_messages else None
                        
                        if latest_ctx_msg and "content" in latest_ctx_msg:
                            formatted_msg["content"] = latest_ctx_msg.get("content", "")
                        elif "content" in msg:
                            formatted_msg["content"] = msg.get("content", "")
                    elif "content" in msg:
                        formatted_msg["content"] = msg.get("content", "")
                    
                    if msg.get("role") == "assistant":
                        formatted_msg["time_to_first_audio"] = self._calculate_time_to_first_audio()
                        formatted_msg["latency"] = {
                            "tts": int(self.metrics.get("BOT_TTS_TIME", 0) * 1000),
                            "llm": int(self.metrics.get("BOT_LLM_TIME", 0) * 1000),
                            "stt": int(self.metrics.get("BOT_STT_TIME", 0) * 1000)
                        }

                        if "tool_calls" in msg:
                            formatted_msg["tool_calls"] = msg["tool_calls"]

                    elif msg.get("role") == "tool_call":
                        formatted_msg["tool_call_id"] = msg.get("tool_call_id", "")
                        formatted_msg["name"] = msg.get("name", "")
                        formatted_msg["arguments"] = msg.get("arguments", {})
                        formatted_msg["function"] = {
                            "name": msg.get("name", ""),
                            "arguments": json.dumps(msg.get("arguments", {}))
                        }

                    elif msg.get("role") == "tool" or msg.get("role") == "tool_result":
                        formatted_msg["role"] = "tool"
                        result = msg.get("result")
                        if result is not None:
                            formatted_msg["content"] = json.dumps(result)
                        else:
                            formatted_msg["content"] = msg.get("content", "")
                        formatted_msg["tool_call_id"] = msg.get("tool_call_id", "")

                    messages.append(formatted_msg)
            
            payload = {
                "sid": self.sid,
                "agent_id": self.agent_id,
                "status": data.get("status", "in_progress"),
                "call_type": "web_call",
                "event_type": event_type,
                "recording_url": "",
                "duration_in_seconds": int(float(data.get("duration", 0))),
                "user_phone_number": "",
                "voice_engine": "Pipecat",
                "created_at": datetime.now().isoformat(),
                "messages": messages,
                "transcriber": self.transcriber,
                "voice_provider": self.voice_provider,
                "llm_model": self.llm_model,
                "llm_provider": self.llm_provider,
            }

            role = data.get("message", {}).get("role", "none") if event_type == "conversation_update" else "none"
            logger.info(f"Sending {event_type} for message role: {role} with data: {payload}")
            if not self.api_key:
                logger.warning("CONNEXITY_API_KEY not found")
                
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            async with aiohttp.request('POST', url, json=payload, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to send message update: {await response.text()}")

        except Exception as e:
            logger.error(f"Error sending message update: {str(e)}")

    def _log_timestamp(self, event_type: str, timestamp: int):
        ts = datetime.fromtimestamp(timestamp / 1_000_000_000)
        self.events[event_type] = {
            'timestamp': timestamp,
            'iso_time': ts.isoformat()
        }

    def _calculate_seconds_from_start(self, msg):
        """Calculate seconds from the start of the conversation."""
        msg_id = id(msg)
        
        if msg_id in self.message_timestamps:
            timestamp = self.message_timestamps[msg_id]
        else:
            timestamp = int(time.time() * 1_000_000_000)
            self.message_timestamps[msg_id] = timestamp
            
        start_timestamp = self.events.get("conversation_start", {}).get("timestamp", timestamp)
        return max(0, (timestamp - start_timestamp) / 1_000_000_000)

    def _extract_context_messages(self, context):
        msgs = []
        for message in context.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "image_url" and item.get("image_url", {}).get("url", "").startswith("data:image/"):
                            item["image_url"]["url"] = "data:image/..."
            if "mime_type" in msg and msg["mime_type"].startswith("image/"):
                msg["data"] = "..."
            msgs.append(msg)
        return msgs
    
    def _get_user_messages_from_context(self):
        """Extract only user messages from context messages."""
        return [msg for msg in self.context_messages if msg.get("role") == "user"]
    
    def _get_assistant_messages_from_context(self):
        """Extract only user messages from context messages."""
        return [msg for msg in self.context_messages if msg.get("role") == "assistant"]
    
    def _find_tool_call_id_from_context(self, function_name):
        """Extract tool_call_id from context messages for a given function name."""
        for msg in self.context_messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tool_call in msg.get("tool_calls", []):
                    if (tool_call.get("type") == "function" and 
                        tool_call.get("function", {}).get("name") == function_name):
                        return tool_call.get("id")
        return None

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        src_name = src.__class__.__name__ if src else "None"
        
        if src and 'TTSService' in src_name and self.voice_provider is None:
            provider_name = src_name.split('TTSService')[0].lower()
            
            provider_map = {
                'elevenlabs': 'elevenlabs',
                'playht': 'playht',
                'cartesia': 'cartesia',
                'lmnt': 'lmnt',
                'openai': 'openai'
            }
            
            self.voice_provider = provider_map.get(provider_name, provider_name)
                
            if hasattr(src, 'model_name'):
                self.transcriber = src.model_name

        if src and 'LLM' in src_name and self.llm_model is None:
            provider_name = src_name.split('LLM')[0]
            self.llm_provider = provider_name
            
            if hasattr(src, 'model_name'):
                self.llm_model = src.model_name
            else:
                self.llm_model = "unknown"      

        if hasattr(frame, 'conversation_id') and frame.conversation_id:
            self.sid = frame.conversation_id
        
        if not self.start_update_sent and not self.stream:
            await self._send_message_update("conversation_update", {
                "status": "in_progress",
                "duration": 0
            })
        
        if isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
            try:
                self.context_messages = self._extract_context_messages(context)
                
                user_messages = self._get_user_messages_from_context()
                if user_messages:
                    latest_user_msg = user_messages[-1]
                    
                    if "content" in latest_user_msg:
                        content = latest_user_msg.get("content", "")
                        
                        if content not in self.sent_message_contents:
                            user_message = {
                                "role": "user",
                                "content": content
                            }
                            
                            self.messages.append(user_message)
                            self.message_timestamps[id(user_message)] = timestamp
                            self.last_user_message_timestamp = timestamp
                            
                            await self._send_message_update("conversation_update", {
                                "status": "in_progress",
                                "duration": self.get_total_duration(),
                                "message": user_message
                            })
                
            except Exception as e:
                logger.error(f"Error extracting context messages: {str(e)}")
                
        elif isinstance(frame, LLMMessagesFrame):
            for msg in frame.messages:
                if msg.get("role") == "system" and not any(m.get("role") == "system" for m in self.messages):
                    self.messages.append(msg)
                    self.message_timestamps[id(msg)] = timestamp
        
        elif isinstance(frame, TextFrame) and src_name == "OpenAILLMService":
            if not self.collecting_assistant_message:
                self.current_assistant_message = ""
                self.collecting_assistant_message = True
            
            self.current_assistant_message += frame.text
        
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._log_timestamp('user_speech_end', timestamp)
            
            if (timestamp - self.last_user_message_timestamp > 1_000_000_000):
                user_messages = self._get_user_messages_from_context()
                latest_context_message = user_messages[-1] if user_messages else None
                
                if latest_context_message and "content" in latest_context_message:
                    content = latest_context_message["content"]
                    if content in self.sent_message_contents:
                        return
                        
                    user_message = {
                        "role": "user",
                        "content": content
                    }
                    
                    self.messages.append(user_message)
                    self.message_timestamps[id(user_message)] = timestamp
                    self.last_user_message_timestamp = timestamp
                    
                    await self._send_message_update("conversation_update", {
                        "status": "in_progress",
                        "duration": self.get_total_duration(),
                        "message": user_message
                    })
        
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._log_timestamp('bot_speech_start', timestamp)
        
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._log_timestamp('bot_speech_end', timestamp)
            
            current_time = timestamp  
            last_bot_message_time = getattr(self, 'last_bot_message_timestamp', 0)
            
            if current_time - last_bot_message_time > 500_000_000:
                if self.collecting_assistant_message and self.current_assistant_message:
                    assistant_message = {
                        "role": "assistant",
                        "content": self.current_assistant_message
                    }
                    
                    self.messages.append(assistant_message)
                    self.message_timestamps[id(assistant_message)] = timestamp
                    self.current_assistant_message = ""
                    self.collecting_assistant_message = False
                    
                    self.last_bot_message_timestamp = current_time
                    
                    if self.stream:
                        await self._send_message_update("conversation_update", {
                            "status": "in_progress",
                            "duration": self.get_total_duration(),
                            "message": assistant_message
                        })

        elif isinstance(frame, FunctionCallInProgressFrame) and src_name == "OpenAILLMService":
            tool_call_id = getattr(frame, 'tool_call_id', None)
            
            if not tool_call_id:
                context_tool_id = self._find_tool_call_id_from_context(frame.function_name)
                if context_tool_id:
                    tool_call_id = context_tool_id
                else:
                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            
            args = frame.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse function arguments: {args}")
                    args = {}
            
            self._current_function = {
                'name': frame.function_name,
                'args': args,
                'start_time': datetime.now().isoformat(),
                'tool_call_id': tool_call_id
            }
        
        elif isinstance(frame, FunctionCallResultFrame) and src_name == "OpenAILLMService":
            frame_id = id(frame)
            if frame_id in self.processed_function_results:
                return
            
            self.processed_function_results.add(frame_id)
            
            if self._current_function:
                tool_call_id = self._current_function.get('tool_call_id')
                
                call_data = {
                    'function': self._current_function['name'],
                    'arguments': self._current_function['args'],
                    'result': frame.result,
                    'start_time': self._current_function['start_time'],
                    'end_time': datetime.now().isoformat(),
                    'tool_call_id': tool_call_id
                }
                self.metrics["FUNCTION_CALLS"].append(call_data)
                
                result_msg = {
                    "role": "tool",
                    "content": json.dumps(frame.result) if isinstance(frame.result, dict) else str(frame.result),
                    "tool_call_id": tool_call_id,
                    "result": frame.result
                }
                
                self.messages.append(result_msg)
                self.message_timestamps[id(result_msg)] = timestamp
                
                self.current_assistant_message = ""
                self.collecting_assistant_message = True
                
                if self.stream:
                    await self._send_message_update("conversation_update", {
                        "status": "in_progress", 
                        "duration": self.get_total_duration(),
                        "message": result_msg
                    })

                self._current_function = None

        elif isinstance(frame, CancelFrame):
            if not self.end_update_sent:
                self.end_update_sent = True
                await self._send_message_update("end_of_call", {
                    "status": "completed",
                    "duration": self.get_total_duration()
                })

        elif isinstance(frame, MetricsFrame):
            if direction == FrameDirection.DOWNSTREAM:
                metrics_data = {}
                for metric in frame.data:
                    metric_key = f"{metric.__class__.__name__}_{metric.processor}"
                    metrics_data[metric_key] = {
                        'processor': metric.processor,
                        'model': metric.model,
                        'value': metric.value
                    }

                    if "OpenAILLMService" in metric.processor and isinstance(metric, TTFBMetricsData):
                        self.metrics["BOT_LLM_TIME"] = metric.value

                    if 'TTSService' in metric.processor and isinstance(metric, TTFBMetricsData):
                        self.metrics["BOT_TTS_TIME"] = metric.value

                    if 'DeepgramSTTService' in metric.processor and isinstance(metric, TTFBMetricsData):
                        self.metrics["BOT_STT_TIME"] = metric.value

                    if 'TTSService' in metric.processor and isinstance(metric, TTSUsageMetricsData):
                        self.metrics["BOT_TTS_CHARACTERS"] += metric.value

                self.events['metrics'] = metrics_data

    def _calculate_time_to_first_audio(self):
        """
        Calculate the time between when the user stopped speaking and when the agent started speaking.
        Returns time in milliseconds.
        """
        if 'user_speech_end' in self.events and 'bot_speech_start' in self.events:
            user_end_ts = self.events['user_speech_end']['timestamp']
            bot_start_ts = self.events['bot_speech_start']['timestamp']
            
            time_to_first_audio = (bot_start_ts - user_end_ts) / 1_000_000
            
            return max(0, int(time_to_first_audio))
        
        return 0
    
    def get_speech_duration(self, speech_type: str) -> float:
        start_key = f'{speech_type}_speech_start'
        end_key = f'{speech_type}_speech_end'
        
        if start_key in self.events and end_key in self.events:
            start_ts = self.events[start_key]['timestamp']
            end_ts = self.events[end_key]['timestamp']
            return (end_ts - start_ts) / 1_000_000_000
        return None
        
    def get_total_duration(self) -> float:
        """Get the total duration of the conversation."""
        start_ts = self.events.get("conversation_start", {}).get("timestamp", 0)
        end_ts = int(time.time() * 1_000_000_000)
        return (end_ts - start_ts) / 1_000_000_000