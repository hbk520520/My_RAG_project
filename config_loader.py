"""
配置中心 —— 整个项目只有一个地方改参数
================================
从 config.yaml 读配置，支持 ${ENV_VAR} 替换敏感信息。
全局单例，所有模块通过 cfg.get("llm", "api_key") 这种点号路径拿值。

技术栈: PyYAML / re (环境变量替换) / pathlib
"""
import os
import re
import yaml
from typing import Any, Dict
from pathlib import Path


class Config:
    """单例配置，进程里只加载一次"""

    _instance = None
    _data: Dict[str, Any] = {}

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()
        # 替换 ${ENV_VAR} 占位符
        raw = self._subst_env(raw)
        self._data = yaml.safe_load(raw)

    @staticmethod
    def _subst_env(text: str) -> str:
        def replacer(m):
            return os.environ.get(m.group(1), "")
        return re.sub(r"\$\{(\w+)\}", replacer, text)

    def get(self, *keys: str, default: Any = None) -> Any:
        """支持点号路径访问，如 cfg.get('llm', 'api_key')"""
        node = self._data
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k)
            else:
                return default
            if node is None:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


# 全局单例
cfg = Config()
