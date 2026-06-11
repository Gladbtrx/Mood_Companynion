"""Phase 4 — 本地夜间 DPO 强化学习管线（模块 E）。

模块划分：
    config.py        共享配置（与 Rust 后端共用 config/backend.json）
    prompts.py       STYLE_PROMPT 的 Python 镜像（须与 style.rs 逐字同步）
    dataset.py       bandit 日志 → advantage 回写 → DPO 偏好对
    train_dpo.py     QLoRA DPO 微调（有 GPU 走真训，否则模拟）+ proxy 评估 + 回滚
    export_ollama.py LoRA → GGUF → ollama create（带版本号）
    scheduler.py     凌晨窗口 + 空闲 + 新样本门控，触发 train_dpo
"""
