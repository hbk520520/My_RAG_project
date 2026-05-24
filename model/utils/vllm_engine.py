"""
自部署推理引擎 —— 用自己的模型跑，用 VLLM 前缀缓存省钱
====================================================
一个通用基座 + 多个 LoRA 模块，VLLM 自动做 prefix caching：
  同一个 system prompt 只算一次 KV cache，后面每次请求重用。

前面的训练产出 .saved_loras/ 里的 LoRA 权重，这里直接挂上去用。
如果 VLLM 没装或显存不够，自动退回 DeepSeek API。

技术栈: vllm (LLM/SamplingParams) / peft / unsloth
"""
import os, sys, logging
from typing import Optional, Dict, Any

logger = logging.getLogger("VLLMEngine")

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


class VLLMPromptCacheEngine:
    """
    用 VLLM 自部署模型，启动时自动开启 prefix caching。
    上游代码不需要改调用方式——对外暴露和 OpenAI 客户端一样的 chat() 接口。
    """

    def __init__(self,
                 base_model: str = "Qwen/Qwen2.5-14B-Instruct",
                 lora_modules: Optional[Dict[str, str]] = None,
                 gpu_memory_utilization: float = 0.85):
        """
        :param base_model:   基座模型名或路径
        :param lora_modules: {"planner": "./saved_loras/planner_lora", ...}
        :param gpu_memory_utilization: VLLM 显存占比
        """
        if not HAS_VLLM:
            raise RuntimeError("vllm 没装：pip install vllm")

        self.base_model = base_model
        self.lora_modules = lora_modules or {}

        # ---- 加载模型（VLLM 自动开启 prefix caching） ----
        logger.info(f"加载基座模型: {base_model}")
        logger.info(f"挂载 LoRA 模块: {list(self.lora_modules.keys())}")

        vllm_kwargs = {
            "model": base_model,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True,   # ← 核心：VLLM 的自动前缀缓存
            "max_model_len": 8192,
            "trust_remote_code": True,
        }

        if self.lora_modules:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64

        self.engine = LLM(**vllm_kwargs)
        logger.info("VLLM 引擎启动完毕，prefix caching 已开启。")

    # ------------------------------------------------------------------
    # 对外接口：和 OpenAI API 同签名
    # ------------------------------------------------------------------
    class FakeResponse:
        """伪造 OpenAI API 返回格式，上游代码不用改"""
        def __init__(self, content: str):
            self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]

    def chat(self,
             messages: list,
             temperature: float = 0.1,
             max_tokens: int = 2048,
             require_json: bool = False,
             lora_name: Optional[str] = None,
             **kwargs) -> "VLLMPromptCacheEngine.FakeResponse":
        """
        和 OpenAI client.chat.completions.create() 同样的参数签名。
        :param lora_name: 指定用哪个 LoRA 适配器。None 则用基座。
        """
        # VLLM 的 chat 接口直接吃 messages 列表
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        # 选择 LoRA 适配器
        lora_request = None
        if lora_name and lora_name in self.lora_modules:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(lora_name, 1, self.lora_modules[lora_name])

        # 调用 VLLM —— prefix caching 在这里自动生效
        outputs = self.engine.chat(
            messages=messages,
            sampling_params=sampling,
            lora_request=lora_request,
        )

        content = outputs[0].outputs[0].text if outputs else ""
        return self.FakeResponse(content)

    # ------------------------------------------------------------------
    # 向下兼容：同时提供和 legal_ops._call_llm 一样的签名
    # ------------------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str,
                  require_json: bool = False, temperature: float = 0.1,
                  lora_name: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.chat(messages, temperature=temperature, lora_name=lora_name)
        return resp.choices[0].message.content.strip()


# ============================================================================
# 工厂函数：自动选择 VLLM 还是 API
# ============================================================================
def create_llm_backend(force: str = "auto") -> Any:
    """
    创建 LLM 后端，自动检测环境。
    force="vllm" / "api" / "auto"
    """
    if force == "api":
        from openai import OpenAI
        from config_loader import cfg
        return OpenAI(api_key=cfg.get("llm", "api_key"), base_url=cfg.get("llm", "base_url"))

    if force == "vllm" or (force == "auto" and HAS_VLLM):
        try:
            base = os.environ.get("VLLM_BASE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
            loras = {
                "planner": "./saved_loras/planner_lora",
                "replanner": "./saved_loras/replanner_grpo",
                "extractor": "./saved_loras/extractor_grader_dpo",
                "reasoner": "./saved_loras/reasoner_sft",
            }
            # 只挂载实际存在的 LoRA
            available = {k: v for k, v in loras.items() if os.path.exists(v)}
            return VLLMPromptCacheEngine(base_model=base, lora_modules=available)
        except Exception as e:
            logger.warning(f"VLLM 启动失败 ({e})，退回 API")
            return create_llm_backend(force="api")

    # 默认：API
    from openai import OpenAI
    from config_loader import cfg
    return OpenAI(api_key=cfg.get("llm", "api_key"), base_url=cfg.get("llm", "base_url"))
