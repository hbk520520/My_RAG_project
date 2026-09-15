"""
配置中心 —— 整个项目只有一个地方改参数
================================
从 config.yaml 读配置，支持两种环境变量占位符：
    ${ENV_VAR}              -> 环境变量，未定义时替换为空串
    ${ENV_VAR:-默认值}       -> 环境变量，未定义或为空时用默认值
全局单例，所有模块通过 cfg.get("llm", "api_key") 这种可变参数路径拿值。

设计要点（阶段 1 新增）：
  1. 配置文件缺失 / YAML 非法 -> 直接抛 ConfigError，不再静默崩在别处
  2. 必填项为空 -> 加载时打一条 WARNING 列出全部缺失项（而不是安静返回空串）
  3. 需要硬失败时调用 cfg.validate(strict=True)

技术栈: PyYAML / re (环境变量替换) / pathlib
"""
import os
import re
import logging
import yaml
from typing import Any, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger("config_loader")

# ${VAR} 或 ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """配置缺失或非法时抛出，避免"静默拿到空串"引发的难排查问题"""


class Config:
    """单例配置，进程里只加载一次"""

    _instance = None
    _data: Dict[str, Any] = {}

    # 必填项：(可变参数路径, 对应环境变量名, 说明)
    REQUIRED: List[Tuple[Tuple[str, ...], str, str]] = [
        (("llm", "api_key"), "DEEPSEEK_API_KEY", "LLM API Key"),
        (("llm", "base_url"), "", "LLM API 地址"),
        (("llm", "judge_model"), "", "默认调用的模型名"),
    ]

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def _load(self, config_path: str = None):
        if config_path is None:
            # 优先环境变量指定的路径，其次取本文件同目录下的 config.yaml
            config_path = os.environ.get("MY_RAG_CONFIG") or (Path(__file__).parent / "config.yaml")

        path = Path(config_path)
        if not path.is_file():
            raise ConfigError(
                f"找不到配置文件: {path}\n"
                f"请确认 config.yaml 存在，或用环境变量 MY_RAG_CONFIG 指定路径。"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError as e:
            raise ConfigError(f"读取配置文件失败: {path} ({e})") from e

        raw = self._subst_env(raw)

        try:
            self._data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"config.yaml 不是合法 YAML: {path} ({e})") from e

        if not isinstance(self._data, dict):
            raise ConfigError(f"config.yaml 顶层必须是映射(dict)，实际是 {type(self._data).__name__}")

        self.config_path = path
        self._warn_missing_required()

    @staticmethod
    def _subst_env(text: str) -> str:
        """把 ${VAR} / ${VAR:-默认值} 替换为实际值"""
        def replacer(m: "re.Match") -> str:
            name, default = m.group(1), m.group(2)
            value = os.environ.get(name)
            if value:                      # 已定义且非空
                return value
            if default is not None:        # ${VAR:-默认值}
                return default
            return ""                      # 只写了 ${VAR} 但没定义 -> 空串
        return _ENV_PATTERN.sub(replacer, text)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def missing_required(self) -> List[str]:
        """返回当前所有缺失的必填项描述"""
        missing = []
        for keys, env_name, desc in self.REQUIRED:
            if not self.get(*keys):
                hint = f"，可通过环境变量 ${{{env_name}}} 提供" if env_name else ""
                missing.append(f"  - {'.'.join(keys)}  ({desc}){hint}")
        return missing

    def _warn_missing_required(self):
        """加载时只告警，不阻断启动 —— 便于本地跑通与容器分阶段注入密钥"""
        missing = self.missing_required()
        if missing:
            logger.warning(
                "配置项为空，相关功能会在首次调用时失败：\n%s\n"
                "（参考 .env.example 设置环境变量后重启即可）",
                "\n".join(missing),
            )

    def validate(self, strict: bool = True):
        """
        显式校验。strict=True 时缺失必填项直接抛 ConfigError。
        启动脚本 / Worker 入口建议调用 cfg.validate() 做快速失败。
        """
        missing = self.missing_required()
        if missing and strict:
            raise ConfigError(
                "配置缺失，请检查 config.yaml 或设置对应环境变量：\n" + "\n".join(missing)
            )
        return not missing

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """按层级取值，如 cfg.get("llm", "api_key")；任一层不存在则返回 default"""
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k)
            if node is None:
                return default
        return node

    def as_dict(self) -> Dict[str, Any]:
        """返回配置的浅拷贝，便于日志 / 调试输出"""
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


# 全局单例
cfg = Config()

