"""
Unsloth 加载器 —— 一张卡也能微调大模型
===================================
4-bit QLoRA 把 14B 模型压到不到 5GB，All-Linear LoRA 挂载，
按模型类型预配 r/alpha 参数。保存只存 LoRA 增量，不存基座。

技术栈: unsloth (FastLanguageModel) / peft (LoRA) / transformers
"""
import os, sys
from typing import Tuple, Optional, List
from unsloth import FastLanguageModel

# 支持 config_loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from config_loader import cfg
    USE_CFG = True
except Exception:
    USE_CFG = False


# ============================================================================
# 预定义模型配置
# ============================================================================
MODEL_CONFIGS = {
    "meta_planner": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "lora_r": 64,
        "lora_alpha": 128,
        "max_seq_length": 4096,
        "description": "Meta-Planner: 抽象推理骨架生成"
    },
    "replanner": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "lora_r": 32,
        "lora_alpha": 64,
        "max_seq_length": 4096,
        "description": "Replanner: 虫洞重规划"
    },
    "extractor_grader": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_length": 2048,
        "description": "Extractor+Grader: 事实提取与充分性判断"
    },
    "reasoner": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "lora_r": 32,
        "lora_alpha": 64,
        "max_seq_length": 8192,
        "description": "Reasoner: 长上下文法律推演"
    },
}


# ============================================================================
# UnslothLoader 类
# ============================================================================
class UnslothLoader:
    """Unsloth 4-bit QLoRA 模型加载器"""

    def __init__(self, model_type: str, custom_model_name: str = None):
        """
        :param model_type: 预定义类型 ("meta_planner"/"replanner"/"extractor_grader"/"reasoner")
                           或任意自定义名称（将从 config.yaml 读取配置）
        :param custom_model_name: 覆盖默认基座模型名
        """
        if model_type in MODEL_CONFIGS:
            self.config = MODEL_CONFIGS[model_type].copy()
        else:
            # 尝试从 config.yaml 读取
            self.config = {
                "base_model": custom_model_name or "Qwen/Qwen2.5-7B-Instruct",
                "lora_r": 32, "lora_alpha": 64, "max_seq_length": 4096,
            }
        if custom_model_name:
            self.config["base_model"] = custom_model_name

    def load(self,
             load_in_4bit: bool = True,
             dtype=None,
             max_seq_length: int = None
             ) -> Tuple[FastLanguageModel, any]:
        """
        加载基座模型并挂载 LoRA。
        :return: (model, tokenizer)
        """
        model_name = self.config["base_model"]
        seq_len = max_seq_length or self.config.get("max_seq_length", 4096)

        print(f"[Unsloth] 加载 {model_name} (4bit={load_in_4bit}, seq_len={seq_len})")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # 挂载 LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config["lora_r"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        return model, tokenizer

    @staticmethod
    def format_chat(examples, tokenizer, text_field: str = "messages") -> dict:
        """将对话格式数据转换为模型输入文本"""
        examples["text"] = tokenizer.apply_chat_template(
            examples[text_field], tokenize=False
        )
        return examples

    @staticmethod
    def save_lora(model, tokenizer, output_dir: str):
        """保存 LoRA 权重（不含基座）"""
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[Unsloth] LoRA 已保存至 {output_dir}")


# ============================================================================
# 便捷函数
# ============================================================================
def load_planner():
    return UnslothLoader("meta_planner").load()


def load_replanner():
    return UnslothLoader("replanner").load()


def load_extractor_grader():
    return UnslothLoader("extractor_grader").load()


def load_reasoner():
    return UnslothLoader("reasoner").load()
