"""
可观测性模块 —— 结构化日志、Metrics、健康检查
"""
import time
import json
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler


# ============================================================================
# 结构化 JSON 日志
# ============================================================================
class JsonFormatter(logging.Formatter):
    """JSON 格式日志输出，方便 ELK/Loki 采集"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            import traceback
            log_entry["exception"] = traceback.format_exception(*record.exc_info)
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging(log_level: str = "INFO", log_format: str = "json"):
    """统一初始化日志系统"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除已有 handler
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
        ))
    root_logger.addHandler(handler)


# ============================================================================
# Prometheus 风格 Metrics 收集器（内存版）
# ============================================================================
class MetricsCollector:
    """线程安全的内存 metrics 收集器，暴露 HTTP 端点"""

    def __init__(self):
        self._lock = threading.Lock()
        # Counter 类型
        self._counters: Dict[str, float] = defaultdict(float)
        # Histogram 类型 (存储所有值)
        self._histograms: Dict[str, list] = defaultdict(list)
        # Gauge 类型
        self._gauges: Dict[str, float] = defaultdict(float)
        # 请求计数时间窗口
        self._window: Dict[str, list] = defaultdict(list)  # (timestamp, value)

    def counter_inc(self, name: str, value: float = 1.0):
        """递增计数器"""
        with self._lock:
            self._counters[name] += value

    def histogram_observe(self, name: str, value: float):
        """记录直方图样本"""
        with self._lock:
            self._histograms[name].append(value)
            if len(self._histograms[name]) > 10000:
                self._histograms[name] = self._histograms[name][-5000:]

    def gauge_set(self, name: str, value: float):
        """设置仪表值"""
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> Dict[str, Any]:
        """获取当前 metrics 快照"""
        with self._lock:
            result = {"counters": {}, "gauges": {}, "histograms": {}}
            result["counters"] = dict(self._counters)
            result["gauges"] = dict(self._gauges)
            for name, values in self._histograms.items():
                if values:
                    vals = sorted(values)
                    result["histograms"][name] = {
                        "count": len(vals),
                        "p50": vals[len(vals) // 2],
                        "p95": vals[int(len(vals) * 0.95)],
                        "p99": vals[int(len(vals) * 0.99)],
                        "min": vals[0],
                        "max": vals[-1],
                    }
            return result


# 全局单例
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics


# ============================================================================
# 健康检查 HTTP 端点
# ============================================================================
class HealthHandler(BaseHTTPRequestHandler):
    """健康检查 + Metrics 端点"""

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "timestamp": time.time()})
        elif self.path == "/ready":
            self._respond(200, {"status": "ready"})
        elif self.path == "/metrics":
            self._respond(200, _metrics.snapshot())
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # 静默 HTTP 日志


def start_health_server(port: int = 8080):
    """在后台线程启动健康检查服务"""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.getLogger("observability").info(f"Health server started on port {port}")
    return server


# ============================================================================
# 指数退避重试装饰器
# ============================================================================
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    指数退避重试装饰器。
    每次重试延迟 = min(base_delay * backoff_factor^attempt, max_delay)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        logger = logging.getLogger("retry")
                        logger.warning(
                            f"{func.__name__} 失败 (attempt {attempt+1}/{max_retries+1}): {e}. "
                            f"{delay:.1f}s 后重试..."
                        )
                        _metrics.counter_inc("retry_attempts")
                        time.sleep(delay)
                    else:
                        logger = logging.getLogger("retry")
                        logger.error(f"{func.__name__} 全部 {max_retries+1} 次重试均失败")
                        _metrics.counter_inc("retry_exhausted")
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# LLM 调用专用：指数退避 + fallback 模型
# ============================================================================
class ResilientLLMClient:
    """
    带容错的 LLM 客户端：指数退避重试 + 主备模型切换。
    """

    def __init__(self,
                 primary_api_key: str,
                 primary_base_url: str,
                 primary_model: str,
                 fallback_api_key: Optional[str] = None,
                 fallback_base_url: Optional[str] = None,
                 fallback_model: Optional[str] = None):
        from openai import OpenAI

        self.primary = OpenAI(api_key=primary_api_key, base_url=primary_base_url)
        self.primary_model = primary_model
        self.fallback = None
        self.fallback_model = fallback_model
        if fallback_api_key and fallback_base_url:
            self.fallback = OpenAI(api_key=fallback_api_key, base_url=fallback_base_url)

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _call_primary(self, messages: list, **kwargs) -> str:
        start = time.time()
        resp = self.primary.chat.completions.create(
            model=self.primary_model,
            messages=messages,
            **kwargs
        )
        elapsed = (time.time() - start) * 1000
        _metrics.histogram_observe("llm_primary_latency_ms", elapsed)
        _metrics.counter_inc("llm_primary_calls")
        return resp.choices[0].message.content.strip()

    def _call_fallback(self, messages: list, **kwargs) -> str:
        if not self.fallback:
            raise RuntimeError("无可用的 fallback 模型")
        start = time.time()
        resp = self.fallback.chat.completions.create(
            model=self.fallback_model,
            messages=messages,
            **kwargs
        )
        elapsed = (time.time() - start) * 1000
        _metrics.histogram_observe("llm_fallback_latency_ms", elapsed)
        _metrics.counter_inc("llm_fallback_calls")
        return resp.choices[0].message.content.strip()

    def chat(self, messages: list, **kwargs) -> str:
        """主备自动切换的 LLM 调用"""
        try:
            return self._call_primary(messages, **kwargs)
        except Exception as e:
            logger = logging.getLogger("llm_resilient")
            logger.warning(f"主模型调用失败: {e}，尝试 fallback...")
            _metrics.counter_inc("llm_fallback_triggered")
            try:
                return self._call_fallback(messages, **kwargs)
            except Exception as fe:
                logger.error(f"Fallback 也失败: {fe}")
                _metrics.counter_inc("llm_total_failure")
                raise


# ============================================================================
# Dead Letter Queue 模拟（基于本地文件）
# ============================================================================
class DeadLetterQueue:
    """
    简易 Dead Letter Queue：将处理失败的消息持久化到本地文件。
    生产环境应切换至 Kafka 专用 DLQ Topic。
    """

    def __init__(self, path: str = "./dlq.jsonl"):
        self.path = path

    def push(self, topic: str, key: str, value: dict, error: str):
        """将失败消息写入 DLQ"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "topic": topic,
            "key": key,
            "value": value,
            "error": str(error)[:500],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.getLogger("dlq").warning(f"消息进入 DLQ: topic={topic}, key={key}")

    def peek(self, limit: int = 100) -> list:
        """查看最近的 DLQ 消息"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [json.loads(line) for line in lines[-limit:]]
        except FileNotFoundError:
            return []

    def replay(self, handler_fn) -> int:
        """
        重放 DLQ 中的所有消息。
        handler_fn(topic, key, value) -> bool (成功返回 True)
        返回成功重放的数量。
        """
        entries = self.peek(limit=10000)
        if not entries:
            return 0

        success_count = 0
        remaining = []
        for entry in entries:
            try:
                ok = handler_fn(entry["topic"], entry["key"], entry["value"])
                if ok:
                    success_count += 1
                else:
                    remaining.append(entry)
            except Exception:
                remaining.append(entry)

        # 重写 DLQ，仅保留仍失败的消息
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in remaining:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logging.getLogger("dlq").info(f"DLQ 重放: {success_count}/{len(entries)} 成功")
        return success_count


# ============================================================================
# LLM 调用计时上下文管理器
# ============================================================================
class LLMTimer:
    """上下文管理器：自动记录 LLM 调用耗时"""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        _metrics.histogram_observe(f"llm_{self.operation}_latency_ms", elapsed)
        _metrics.counter_inc(f"llm_{self.operation}_calls")
