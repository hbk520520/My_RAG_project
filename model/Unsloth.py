"""
Unsloth 4-bit QLoRA 快速上手
============================
本文件只是一段「最短路径」示范脚本，**不含任何独立配置**。

阶段 7 变更：
  原先这份脚本自己内联了一整套 QLoRA 参数（r=64 / lora_alpha=128 / target_modules=…），
  并把基座写死成 Qwen/Qwen2.5-7B-Instruct。而 config.yaml 的
  training.planner.base_model 是 Qwen2.5-14B-Instruct，model/utils/unsloth_loader.py
  的 MODEL_CONFIGS["meta_planner"] 也是 14B —— 三方不一致，且改一处不会同步另外两处。
  现在统一委托 UnslothLoader：参数只有一处真源（MODEL_CONFIGS / config.yaml）。

真正的训练入口是 model/training/*.py，本文件仅用于「先跑通加载链路」验证环境。

依赖：unsloth 需要 CUDA 环境，本项目的 law_rag conda 环境未安装（也不该安装），
      因此必须在 GPU 机器上执行。安装方式见 README「模型训练策略」。

运行：
    python -m model.Unsloth        # 只加载 + 挂 LoRA，不训练
"""
import os
import sys

# 路径引导：让 model/utils 与仓库根目录（config_loader）可被导入
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "utils"), _HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from unsloth_loader import UnslothLoader
except ImportError as e:  # unsloth 未安装（无 GPU 环境）时的友好提示
    raise SystemExit(
        "未安装 unsloth，本脚本无法运行。\n"
        "  Unsloth 需要 CUDA 环境，本项目 law_rag conda 环境未安装它；\n"
        "  请到 GPU 机器上执行模型训练（见 README「模型训练策略」）。\n"
        f"  原始错误：{e}"
    ) from e


def main():
    # 关键点：r / lora_alpha / max_seq_length / base_model 全部来自 UnslothLoader，
    # 本脚本不再重复声明，避免与 config.yaml 漂移。
    loader = UnslothLoader("meta_planner")
    c = loader.config
    print(f"[Quickstart] 基座={c['base_model']}  r={c['lora_r']}  "
          f"alpha={c['lora_alpha']}  max_seq_length={c['max_seq_length']}")
    print(f"[Quickstart] 用途={c.get('description', '-')}")

    model, tokenizer = loader.load()
    print("[Quickstart] 4-bit QLoRA 加载完成，LoRA 已挂载（All-Linear）。")
    print("[Quickstart] 正式训练请执行：python -m model.training.train_meta_planner")
    return model, tokenizer


if __name__ == "__main__":
    main()
