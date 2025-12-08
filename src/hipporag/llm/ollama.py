import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import ollama  # <--- 關鍵改變：改用 ollama 套件
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_fixed

# 假設這些 utils 你原本的專案都有，直接引用即可
from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

# ==========================================
# Decorator 1: 快取功能 (完全不用改，邏輯通用的)
# ==========================================
def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1. 抓取 messages
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # 2. 抓取參數
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # 3. 製作指紋 (Hash Key)
        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        # 4. 嘗試從 SQLite 讀取
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                return message, metadata, True # Hit!

        # 5. 沒讀到，執行原本的函數 (Call LLM)
        result = func(self, *args, **kwargs)
        message, metadata = result

        # 6. 寫入 SQLite
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False # Miss

    return wrapper

# ==========================================
# Decorator 2: 自動重試 (完全不用改)
# ==========================================
def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

# ==========================================
# Class: CacheOllama (核心改動都在這)
# ==========================================
class OllamaLLM(BaseLLM):
    """Ollama LLM implementation with Caching."""
    
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "OllamaLLM":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = getattr(global_config, 'max_retry_attempts', 3)
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None, **kwargs) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name  # 例如: "llama3.1"
        
        # 這裡我們要指定 162 的位置，通常 global_config 會有 llm_base_url
        # 如果沒有，預設就是 162
        self.llm_base_url = getattr(global_config, 'llm_base_url', 'http://140.122.184.162:11434')

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            # 檔名改成 ollama_cache
            safe_name = self.llm_name.replace('/', '_').replace(':', '_')
            cache_filename = f"ollama_{safe_name}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        
        self.max_retries = kwargs.get("max_retries", 2)

        # 建立 Ollama Client (連線到 162)
        logger.info(f"Connecting to Ollama at {self.llm_base_url}")
        self.client = ollama.Client(host=self.llm_base_url)

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        
        # 設定 Ollama 專用的參數
        # 注意: Ollama 的參數通常叫做 num_predict 而不是 max_tokens
        config_dict['generate_params'] = {
                "model": self.llm_name,
                "options": {
                    "num_predict": config_dict.get("max_new_tokens", 400),
                    "temperature": config_dict.get("temperature", 0.0),
                    "seed": config_dict.get("seed", 0),
                    # 其他可以加 top_p, top_k 等等
                }
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        
        # 1. 準備參數
        # 我們要把 messages 和 options 分開處理
        params = deepcopy(self.llm_config.generate_params)
        
        # 如果 kwargs 有傳入參數，覆蓋原本的設定
        if kwargs:
            # 如果是 model 就直接覆蓋
            if "model" in kwargs:
                params["model"] = kwargs["model"]
            
            # 如果是參數，要塞進 options 裡面
            for key in ["temperature", "seed", "max_new_tokens"]:
                if key in kwargs:
                    # 對應 Ollama 的參數名
                    ollama_key = "num_predict" if key == "max_new_tokens" else key
                    params["options"][ollama_key] = kwargs[key]

        logger.debug(f"Calling Ollama API with model: {params['model']}")

        # 2. 呼叫 Ollama (這是跟 OpenAI 最大的不同)
        # Ollama 用 chat 方法，並且把參數放在 options 字典裡
        response = self.client.chat(
            model=params['model'],
            messages=messages,
            options=params['options']
        )

        # 3. 整理結果
        # Ollama 回傳的是一個字典，結構稍微不一樣
        response_message = response['message']['content']
        
        # 4. 整理 Metadata (統計數據)
        metadata = {
            "prompt_tokens": response.get('prompt_eval_count', 0), 
            "completion_tokens": response.get('eval_count', 0),
            "finish_reason": response.get('done_reason', 'stop'),
            "total_duration": response.get('total_duration', 0)
        }

        return response_message, metadata