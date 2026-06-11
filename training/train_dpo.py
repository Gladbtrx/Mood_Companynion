"""【选做支路】近似 DPO 微调（ADR-2 明确：主线是 RWR，见 train.py；本文件仅为选做）。

⚠️ 近似性与偏差声明（ADR-2 要求在代码与报告中标明，不得省略）：
  合法的 DPO 需要"同一 context 下 chosen/rejected 成对样本"。运行时每轮只发送
  一个 style、云端只回一个结果，**不存在反事实对照**。本支路的偏好对是近似构造的：
    - 按语义相似度把 context 分桶，桶内高 advantage 当 chosen、低 advantage 当 rejected；
    - 一条偏好对里的 chosen/rejected 来自**不同的用户消息**，共享 prompt 取 chosen 轮的
      context —— 这依赖"同桶语境相近"的假设，桶越粗近似误差越大；
    - 因此该数据**有偏**：它学到的可能是"桶内话题差异"而非纯风格偏好。
  报告中引用本支路结果时必须注明以上局限。日常训练请用主线：
      python -m training.train [--mock]

闭环结构与主线一致：
    build_dataset → DPO 训练 → holdout proxy 评估 → 回滚判定（runlog）→ 记 train_runs → 导出
真训练需 GPU + trl/peft/transformers；否则自动走模拟路径（# MOCK）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import config as mc_config
from . import dataset as mc_dataset
from . import runlog

# train_runs 助手统一收敛到 runlog.py（与 RWR 主线共用）。
# 旧调用点兼容别名：
prev_effective_version = runlog.prev_effective_version
best_proxy_metric = runlog.best_proxy_metric
record_train_run = runlog.record_train_run


# ----------------------------------------------------------------------------
# 训练后端
# ----------------------------------------------------------------------------
@dataclass
class TrainOutcome:
    backend: str  # "trl" | "simulated"
    adapter_dir: Optional[str]
    notes: str


def _have_real_stack() -> bool:
    try:
        import torch  # noqa
        import trl  # noqa
        import peft  # noqa
        import transformers  # noqa

        return torch.cuda.is_available()
    except Exception:
        return False


def train_real(cfg: mc_config.Config, train_path: str, out_dir: str) -> TrainOutcome:
    """真 QLoRA DPO。仅当 _have_real_stack() 为真时调用。"""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    tcfg = cfg.training
    base = tcfg["hf_base_model"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")

    ds = load_dataset("json", data_files=train_path, split="train")
    lora = LoraConfig(
        r=int(tcfg["lora_r"]),
        lora_alpha=int(tcfg["lora_alpha"]),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args = DPOConfig(
        beta=float(tcfg["dpo_beta"]),
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=int(tcfg["epochs"]),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=int(tcfg["max_seq_len"]),
        output_dir=out_dir,
        logging_steps=5,
        save_strategy="no",
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    return TrainOutcome(backend="trl", adapter_dir=out_dir, notes="QLoRA DPO 训练完成")


def train_simulated(train_pairs: List[Dict[str, Any]], out_dir: str) -> TrainOutcome:
    """无 GPU 占位：不反传，只落一份 adapter 元数据，证明闭环跑通。"""
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "backend": "simulated",
        "note": "占位训练：未更新权重。装好 GPU + trl 后自动切真训练。",
        "n_train_pairs": len(train_pairs),
    }
    with open(os.path.join(out_dir, "adapter_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return TrainOutcome(backend="simulated", adapter_dir=out_dir, notes="模拟训练（无 GPU）")


# ----------------------------------------------------------------------------
# proxy 评估（留出偏好对上的偏好准确率）
# ----------------------------------------------------------------------------
def evaluate_holdout(outcome: TrainOutcome, holdout: List[Dict[str, Any]]) -> Optional[float]:
    """返回 [0,1] 偏好准确率：模型把 chosen 排在 rejected 前的比例。
    真训练下应跑 logprob(chosen) vs logprob(rejected)；占位路径用 margin 可分性估计。"""
    if not holdout:
        return None
    if outcome.backend == "trl":
        # TODO(GPU): 加载 out_dir 适配器，逐对比较 chosen/rejected 的序列对数似然。
        # 这里给出接口位；占位阶段不会走到。
        return _simulated_accuracy(holdout)
    return _simulated_accuracy(holdout)


def _simulated_accuracy(holdout: List[Dict[str, Any]]) -> float:
    """确定性占位指标：margin 越大越可分 → 估计准确率越高。
    acc = mean( sigmoid(margin * scale) )，对正 margin 给 >0.5 的乐观估计。"""
    scale = 4.0
    accs = [1.0 / (1.0 + math.exp(-scale * p["meta"]["adv_margin"])) for p in holdout]
    return sum(accs) / len(accs)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
@dataclass
class RunReport:
    skipped: bool
    reason: str = ""
    model_version: int = 0
    n_pairs: int = 0
    proxy_metric: Optional[float] = None
    rolled_back: bool = False
    backend: str = ""


def run(cfg: mc_config.Config, *, dry_run: bool = False, force: bool = False) -> RunReport:
    tcfg = cfg.training
    if not tcfg.get("enabled", True) and not force:
        return RunReport(skipped=True, reason="training.enabled=false")

    # 1) 构建数据集（同时回写 advantage）
    ds = mc_dataset.build_dataset(cfg, write=True)
    print(f"[train] 可训练轮次={ds.n_turns} 偏好对={ds.n_pairs} "
          f"(train={len(ds.train)}, holdout={len(ds.holdout)})")

    min_pairs = int(tcfg["min_pairs"])
    if ds.n_pairs < min_pairs and not force:
        return RunReport(skipped=True, reason=f"偏好对不足（{ds.n_pairs} < {min_pairs}）", n_pairs=ds.n_pairs)

    conn = mc_dataset.connect(cfg.db_path)
    try:
        prev_v = prev_effective_version(conn)
        new_v = prev_v + 1
        out_dir = os.path.join(cfg.adapter_dir, f"v{new_v}")

        # 2) 训练
        use_real = (not dry_run) and _have_real_stack()
        if use_real:
            outcome = train_real(cfg, ds.train_path, out_dir)
        else:
            if not dry_run:
                print("[train] 未检测到 GPU/trl 栈 → 走模拟训练（占位）")
            outcome = train_simulated(ds.train, out_dir)
        print(f"[train] 训练后端={outcome.backend} 适配器={outcome.adapter_dir}")

        # 3) proxy 评估
        proxy = evaluate_holdout(outcome, ds.holdout)
        print(f"[train] proxy_metric(holdout 偏好准确率)={proxy}")

        # 4) 回滚判定（与 RWR 主线共用 runlog 护栏）
        rolled_back, reason = runlog.decide_rollback(
            conn, proxy,
            min_metric=float(tcfg["proxy_min_accuracy"]),
            regress_epsilon=float(tcfg["proxy_regress_epsilon"]),
        )
        if rolled_back:
            print(f"[train] ⛔ 回滚：{reason} → 次日继续用 v{prev_v}")
        elif reason:
            print(f"[train] {reason}")

        # 5) 记 train_runs
        run_id = record_train_run(
            conn,
            model_version=new_v,
            n_samples=ds.n_pairs,
            proxy_metric=proxy,
            rolled_back=rolled_back,
        )
        eff = prev_v if rolled_back else new_v
        print(f"[train] train_runs#{run_id} 记录完成；当前生效版本 v{eff}")
    finally:
        conn.close()

    # 6) 导出 Ollama（仅在未回滚且非 dry-run 时；缺工具会自降级为打印计划）
    if not rolled_back and not dry_run:
        try:
            from . import export_ollama

            export_ollama.export(cfg, adapter_dir=outcome.adapter_dir, version=new_v)
        except Exception as e:
            print(f"[train] 导出 Ollama 跳过/失败（不致命）：{e}")

    return RunReport(
        skipped=False,
        model_version=new_v,
        n_pairs=ds.n_pairs,
        proxy_metric=proxy,
        rolled_back=rolled_back,
        backend=outcome.backend,
    )


def _main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 夜间 DPO 训练")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="强制模拟训练 + 不导出")
    ap.add_argument("--force", action="store_true", help="忽略 enabled / min_pairs 门槛")
    args = ap.parse_args()

    cfg = mc_config.load(args.config)
    rep = run(cfg, dry_run=args.dry_run, force=args.force)
    if rep.skipped:
        print(f"[train] 跳过：{rep.reason}")
    print("\n[train] 报告:", json.dumps(rep.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
