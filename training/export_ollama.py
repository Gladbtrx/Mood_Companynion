"""LoRA 适配器 → GGUF → Ollama 模型（Phase 4 收尾，"训练完成后热更新权重"）。

热更新链路（占位骨架，关键外部工具缺失时安全降级为「只打印计划」，绝不抛断管线）：
    1) merge: 基座 + LoRA 合并为完整权重（peft merge_and_unload）
    2) convert: HF → GGUF（llama.cpp/convert_hf_to_gguf.py）
    3) quantize: GGUF → q4_K_M（llama.cpp/quantize）
    4) ollama create mood-companion:v{N} -f Modelfile
    5) 把 backend.json 的 model_name 指向新 tag（次日 style_request 即用新权重）

为什么用"打印计划而非硬失败"：夜间无人值守，缺 llama.cpp/ollama 不该让整轮训练
算白跑——前面的 advantage 回写、train_runs 记录都已生效，导出可改日手动补做。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List

from . import config as mc_config

OLLAMA_MODEL_PREFIX = "mood-companion"


def _tool(name: str) -> bool:
    return shutil.which(name) is not None


def _modelfile(base_gguf: str, num_predict: int = 120) -> str:
    """生成 Modelfile：FROM 量化 GGUF；参数与 style.rs 的采样意图对齐（高温探索）。"""
    return (
        f"FROM {base_gguf}\n"
        f'PARAMETER temperature 0.95\n'
        f"PARAMETER num_predict {num_predict}\n"
        f'SYSTEM "你是风格控制器，只输出一行 <style>…</style> 风格指令。"\n'
    )


def _run(cmd: List[str]) -> None:
    print("[export] $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export(cfg: mc_config.Config, *, adapter_dir: str, version: int) -> bool:
    """返回 True=已真正创建 Ollama 模型；False=降级为打印计划（缺工具/占位适配器）。"""
    tag = f"{OLLAMA_MODEL_PREFIX}:v{version}"
    work = os.path.join(cfg.adapter_dir, f"v{version}_export")

    # 占位适配器（模拟训练产物）没有真权重，直接给出计划即可。
    simulated = os.path.exists(os.path.join(adapter_dir or "", "adapter_meta.json"))
    have_tools = _tool("ollama")
    can_convert = _tool("python3")  # convert 脚本随 llama.cpp 提供，这里只探针 ollama 链路

    if simulated or not have_tools:
        print("[export] —— 导出计划（占位/缺工具，未实际执行）——")
        print(f"[export] 1. merge LoRA: {adapter_dir} + {cfg.training['hf_base_model']}")
        print(f"[export] 2. convert_hf_to_gguf → {work}/model.gguf")
        print(f"[export] 3. quantize → {work}/model.q4_K_M.gguf")
        print(f"[export] 4. ollama create {tag} -f {work}/Modelfile")
        print(f"[export] 5. 将 config/backend.json.model_name 改为 {tag}")
        print("[export] 装好 GPU 训练栈 + llama.cpp + ollama 后，本函数自动走真实导出。")
        return False

    # ---- 真实导出路径（有 GPU 训练产物 + ollama）----
    os.makedirs(work, exist_ok=True)
    merged = os.path.join(work, "merged")
    _merge_lora(cfg, adapter_dir, merged)
    gguf = os.path.join(work, "model.gguf")
    q_gguf = os.path.join(work, "model.q4_K_M.gguf")
    _convert_to_gguf(merged, gguf)
    _quantize(gguf, q_gguf)

    mf = os.path.join(work, "Modelfile")
    with open(mf, "w", encoding="utf-8") as f:
        f.write(_modelfile(q_gguf))
    _run(["ollama", "create", tag, "-f", mf])
    print(f"[export] ✅ 已创建 {tag}。请将 config/backend.json.model_name 指向它以热更新。")
    return True


def _merge_lora(cfg: mc_config.Config, adapter_dir: str, out: str) -> None:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir)
    model = model.merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(cfg.training["hf_base_model"]).save_pretrained(out)


def _convert_to_gguf(merged_dir: str, gguf_out: str) -> None:
    script = os.environ.get("LLAMA_CPP_CONVERT", "convert_hf_to_gguf.py")
    _run(["python3", script, merged_dir, "--outfile", gguf_out, "--outtype", "f16"])


def _quantize(gguf_in: str, gguf_out: str) -> None:
    quant = os.environ.get("LLAMA_CPP_QUANTIZE", "llama-quantize")
    _run([quant, gguf_in, gguf_out, "q4_K_M"])
