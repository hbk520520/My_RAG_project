"""One bounded completion interface for OpenAI-compatible and local backends."""
from __future__ import annotations

import json
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from config_loader import cfg
from agent_runtime.context import current_run, tool_call


def create_client(api_key=None, base_url=None):
    backend = cfg.get("llm", "backend", default="api")
    if backend == "vllm":
        from model.utils.vllm_engine import VLLMPromptCacheEngine
        return VLLMPromptCacheEngine(
            base_model=cfg.get("vllm", "base_model"),
            lora_modules=cfg.get("vllm", "lora_modules", default={}),
            gpu_memory_utilization=cfg.get("vllm", "gpu_memory_utilization", default=0.85))
    if backend not in ("api", "auto"):
        raise ValueError(f"Unknown LLM backend: {backend}")
    return OpenAI(api_key=api_key or cfg.get("llm", "api_key"),
                  base_url=base_url or cfg.get("llm", "base_url"),
                  timeout=cfg.get("llm", "request_timeout", default=30), max_retries=0)


def complete(client, model, messages, *, require_json=False, temperature=0.1):
    max_output = cfg.get("llm", "max_tokens_default", default=2048)
    context = current_run.get()

    def invoke():
        for attempt in range(cfg.get("llm", "transport_retries", default=1) + 1):
            if context:
                context.check()
                context.model_budget(messages, max_output)
            try:
                if callable(client.chat):
                    response = client.chat(messages, temperature=temperature, max_tokens=max_output,
                                           require_json=require_json)
                else:
                    timeout = cfg.get("llm", "request_timeout", default=30)
                    if context:
                        timeout = min(timeout, max(0.01, context.store.get(context.run_id)["deadline"] - time.time()))
                    response = client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature,
                        max_tokens=max_output, timeout=timeout,
                        response_format={"type": "json_object"} if require_json else None)
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Model returned empty content")
                if require_json:
                    json.loads(content)
                return content.strip()
            except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
                retryable = not isinstance(exc, APIStatusError) or exc.status_code == 429 or exc.status_code >= 500
                if not retryable or attempt == cfg.get("llm", "transport_retries", default=1):
                    raise
                if context:
                    context.check()
                time.sleep(min(0.25 * 2 ** attempt, 1))

    return tool_call("llm.complete", {"model": model, "messages": messages,
                    "json": require_json, "temperature": temperature}, invoke)
