"""
训练层子包。

目录职责（本文件为显式包声明，避免 "model" 这个通用名被命名空间包语义混淆）：
  data/       训练数据工厂（从真实法条反向出题，产 Ground Truth / 高噪点测试集）
  training/   训练脚本（SFT / DPO / GRPO）
  utils/      加载与推理工具（Unsloth 4-bit QLoRA、vLLM 引擎）
  Unsloth.py  4-bit QLoRA 最短路径示范（委托 utils.unsloth_loader，无独立配置）

入口：
  python -m model.Unsloth
  python -m model.training.train_meta_planner
  python -m model.data.evol_instruct

注意：本包依赖 GPU 侧第三方库（unsloth / trl / peft / vllm），
      不在 law_rag 运行时环境中安装。
"""
