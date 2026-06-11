"""夜间 RWR 训练主线（模块 E / ADR-2 / 阶段 4 的 `train.py`）。

为什么主线是 RWR 而不是 DPO（ADR-2，不得擅自推翻）：
  运行时每轮只发送一个 style、云端只回一个结果，拿不到"同一 context 下
  chosen/rejected"的反事实对照，凑不出合法 DPO 数据对。RWR（奖励加权回归）
  直接消费运行时真实产生的 (context x, style s, reward r) 三元组：
  把高 advantage 的 style 当作高权重的 SFT 目标，免成对数据、与标量行为奖励天然契合。

      样本权重 w = clip( exp(advantage / beta), 0, w_max )        —— 第 6 节

完整回环：
    读 SQLite(NORMAL, excluded=0) → 分桶 + advantage 回写 → 加权样本
      → QLoRA(4-bit) 奖励加权 SFT（无 GPU / --mock 时走模拟）
      → 留出集代理指标 → 不升则回滚（runlog.decide_rollback）
      → 记 train_runs → merge → GGUF → ollama create（export_ollama，缺工具自动降级）

DPO 仅作为选做支路（train_dpo.py），是近似、有偏的构造，见该文件头注释。

用法：
    python -m training.train            # 正常（有 GPU+HF 栈走真训，否则自动模拟）
    python -m training.train --mock     # 强制模拟训练 + 不导出（阶段 4 验收用）
    python -m training.train --force    # 忽略 enabled / min_samples 门槛
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import config as mc_config
from . import dataset as mc_dataset
from . import prompts as mc_prompts
from . import runlog


# ----------------------------------------------------------------------------
# 加权样本构造（第 6 节：w = clip(exp(advantage/beta), 0, w_max)）
# ----------------------------------------------------------------------------
def rwr_weight(advantage: float, beta: float, w_max: float) -> float:
    return max(0.0, min(math.exp(advantage / beta), w_max))


def build_rwr_samples(cfg: mc_config.Config, *, write: bool = True) -> Dict[str, Any]:
    """日志 → 分桶 → advantage（回写 DB，dry-run 除外）→ 加权 SFT 样本 + 留出集。

    样本格式（JSONL 一行一条）：
      { "prompt": <运行时同款 style-prompt>, "completion": <style_payload>,
        "weight": w, "meta": {...} }
    """
    tcfg = cfg.training
    beta = float(tcfg["rwr_beta"])
    w_max = float(tcfg["rwr_w_max"])

    conn = mc_dataset.connect(cfg.db_path)
    try:
        turns = mc_dataset.fetch_trainable_turns(conn)
        mc_dataset.assign_buckets(turns, tcfg["bucketer"])
        baselines = mc_dataset.compute_and_write_advantage(conn, turns, write_back=write)
    finally:
        conn.close()

    persona_cache: Dict[str, Dict[str, Any]] = {}
    samples: List[Dict[str, Any]] = []
    for t in turns:
        if t.persona_id not in persona_cache:
            persona_cache[t.persona_id] = mc_prompts.load_persona(cfg.persona_dir, t.persona_id)
        w = rwr_weight(t.advantage, beta, w_max)
        if w <= 1e-6:
            continue  # 权重截断后为 0 的样本（强负 advantage）直接丢弃
        samples.append(
            {
                "prompt": mc_prompts.render_style_prompt(persona_cache[t.persona_id], t.user_input),
                "completion": t.style_payload,
                "weight": round(w, 6),
                "meta": {
                    "turn_id": t.id,
                    "bucket": t.bucket,
                    "reward": round(t.reward, 4),
                    "advantage": round(t.advantage, 4),
                },
            }
        )

    train, holdout = _split_holdout(samples, float(tcfg["holdout_frac"]))
    train_path = os.path.join(cfg.dataset_dir, "rwr_train.jsonl")
    holdout_path = os.path.join(cfg.dataset_dir, "rwr_holdout.jsonl")
    if write:
        mc_dataset.write_jsonl(train_path, train)
        mc_dataset.write_jsonl(holdout_path, holdout)

    return {
        "n_turns": len(turns),
        "samples": samples,
        "train": train,
        "holdout": holdout,
        "baselines": baselines,
        "train_path": train_path,
        "holdout_path": holdout_path,
    }


def _split_holdout(samples: List[Dict[str, Any]], frac: float) -> Tuple[List, List]:
    """确定性切分（按 turn_id 取模，可复现、无随机源）。"""
    if frac <= 0 or len(samples) < 5:
        return samples, []
    mod = max(2, round(1 / frac))
    train = [s for s in samples if s["meta"]["turn_id"] % mod != 0]
    hold = [s for s in samples if s["meta"]["turn_id"] % mod == 0]
    return (samples, []) if not train else (train, hold)


# ----------------------------------------------------------------------------
# 训练后端：真 QLoRA 加权 SFT / 模拟
# ----------------------------------------------------------------------------
@dataclass
class TrainOutcome:
    backend: str  # "hf" | "simulated"
    adapter_dir: Optional[str]
    notes: str


def _have_real_stack() -> bool:
    try:
        import torch  # noqa
        import peft  # noqa
        import transformers  # noqa

        return torch.cuda.is_available()
    except Exception:
        return False


def train_real(cfg: mc_config.Config, train_path: str, out_dir: str) -> TrainOutcome:
    """真 QLoRA 奖励加权 SFT（transformers+peft+bitsandbytes，ADR-5）。
    加权方式：自定义 Trainer，把每条样本的 loss 乘以其 RWR 权重。"""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    tcfg = cfg.training
    base = tcfg["hf_base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(tcfg["lora_r"]),
            lora_alpha=int(tcfg["lora_alpha"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    max_len = int(tcfg["max_seq_len"])

    def tokenize(ex):
        # 只对 completion 计 loss：prompt 部分 labels=-100
        p = tok(ex["prompt"], truncation=True, max_length=max_len)
        full = tok(ex["prompt"] + ex["completion"] + tok.eos_token, truncation=True, max_length=max_len)
        labels = list(full["input_ids"])
        labels[: len(p["input_ids"])] = [-100] * min(len(p["input_ids"]), len(labels))
        return {"input_ids": full["input_ids"], "attention_mask": full["attention_mask"],
                "labels": labels, "weight": ex["weight"]}

    ds = load_dataset("json", data_files=train_path, split="train").map(
        tokenize, remove_columns=["prompt", "completion", "meta"]
    )

    class WeightedTrainer(Trainer):
        """RWR 核心：per-sample loss × clip(exp(advantage/beta), 0, w_max)。"""

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            weights = inputs.pop("weight")
            outputs = model(**inputs)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = inputs["labels"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            tok_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            tok_loss = tok_loss.view(labels.size(0), -1)
            mask = (labels != -100).float()
            per_sample = (tok_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = (per_sample * weights.to(per_sample.dtype)).mean()
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=int(tcfg["epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    WeightedTrainer(model=model, args=args, train_dataset=ds, tokenizer=tok).train()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    return TrainOutcome(backend="hf", adapter_dir=out_dir, notes="QLoRA 加权 SFT 完成")


def train_mock(train: List[Dict[str, Any]], out_dir: str) -> TrainOutcome:
    """--mock / 无 GPU 占位：不反传，落一份 adapter 元数据，把回环串起来（阶段 4 验收）。"""
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "backend": "simulated",
        "note": "MOCK 训练：未更新权重。装好 GPU + transformers/peft/bitsandbytes 后自动切真训练。",
        "n_train_samples": len(train),
        "total_weight": round(sum(s["weight"] for s in train), 4),
    }
    with open(os.path.join(out_dir, "adapter_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return TrainOutcome(backend="simulated", adapter_dir=out_dir, notes="模拟训练（mock）")


# ----------------------------------------------------------------------------
# 代理指标（模块 E"平均 advantage"口径）
# ----------------------------------------------------------------------------
def evaluate_holdout(outcome: TrainOutcome, holdout: List[Dict[str, Any]]) -> Optional[float]:
    """留出集代理指标 ∈ [0,1]：加权平均 advantage 过 sigmoid。
    真训练下应改为：新旧权重在留出 prompt 上生成 style，比较其落在
    高 advantage 桶的比例（风格分类器命中率）。# TODO(GPU)：接真实生成评估。"""
    if not holdout:
        return None
    tw = sum(s["weight"] for s in holdout)
    if tw <= 0:
        return None
    wadv = sum(s["weight"] * s["meta"]["advantage"] for s in holdout) / tw
    return 1.0 / (1.0 + math.exp(-4.0 * wadv))


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
@dataclass
class RunReport:
    skipped: bool
    reason: str = ""
    model_version: int = 0
    n_samples: int = 0
    proxy_metric: Optional[float] = None
    rolled_back: bool = False
    backend: str = ""


def run(cfg: mc_config.Config, *, mock: bool = False, force: bool = False) -> RunReport:
    tcfg = cfg.training
    if not tcfg.get("enabled", True) and not force:
        return RunReport(skipped=True, reason="training.enabled=false")

    # 1) 加权样本（同时回写 advantage；mock 也回写——这是真实的派生数据，非训练副作用）
    data = build_rwr_samples(cfg, write=True)
    n = len(data["samples"])
    print(f"[rwr] 可训练轮次={data['n_turns']} 加权样本={n} "
          f"(train={len(data['train'])}, holdout={len(data['holdout'])})")

    min_samples = int(tcfg["min_samples"])
    if n < min_samples and not force:
        return RunReport(skipped=True, reason=f"样本不足（{n} < {min_samples}）", n_samples=n)

    conn = mc_dataset.connect(cfg.db_path)
    try:
        prev_v = runlog.prev_effective_version(conn)
        new_v = prev_v + 1
        out_dir = os.path.join(cfg.adapter_dir, f"v{new_v}")

        # 2) 训练
        if not mock and _have_real_stack():
            outcome = train_real(cfg, data["train_path"], out_dir)
        else:
            if not mock:
                print("[rwr] 未检测到 GPU/HF 栈 → 走模拟训练（# MOCK）")
            outcome = train_mock(data["train"], out_dir)
        print(f"[rwr] 训练后端={outcome.backend} 适配器={outcome.adapter_dir}")

        # 3) 代理指标 + 回滚护栏
        proxy = evaluate_holdout(outcome, data["holdout"])
        print(f"[rwr] proxy_metric(留出集加权 advantage)={proxy}")
        rolled_back, why = runlog.decide_rollback(
            conn, proxy,
            min_metric=float(tcfg["proxy_min_accuracy"]),
            regress_epsilon=float(tcfg["proxy_regress_epsilon"]),
        )
        if rolled_back:
            print(f"[rwr] ⛔ 回滚：{why} → 次日继续用 v{prev_v}")
        elif why:
            print(f"[rwr] {why}")

        # 4) 记 train_runs
        run_id = runlog.record_train_run(
            conn, model_version=new_v, n_samples=n, proxy_metric=proxy, rolled_back=rolled_back
        )
        print(f"[rwr] train_runs#{run_id} 已记录；当前生效版本 v{prev_v if rolled_back else new_v}")
    finally:
        conn.close()

    # 5) 回环收尾：merge → GGUF → ollama create（缺工具自动降级为打印计划）
    if not rolled_back and not mock:
        try:
            from . import export_ollama

            export_ollama.export(cfg, adapter_dir=outcome.adapter_dir, version=new_v)
        except Exception as e:
            print(f"[rwr] 导出 Ollama 跳过/失败（不致命）：{e}")

    return RunReport(
        skipped=False, model_version=new_v, n_samples=n,
        proxy_metric=proxy, rolled_back=rolled_back, backend=outcome.backend,
    )


def _main() -> int:
    ap = argparse.ArgumentParser(description="夜间 RWR 训练主线（ADR-2）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="强制模拟训练 + 不导出（阶段 4 验收路径）")
    ap.add_argument("--force", action="store_true", help="忽略 enabled / min_samples 门槛")
    args = ap.parse_args()

    cfg = mc_config.load(args.config)
    rep = run(cfg, mock=args.mock, force=args.force)
    if rep.skipped:
        print(f"[rwr] 跳过：{rep.reason}")
    print("\n[rwr] 报告:", json.dumps(rep.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
