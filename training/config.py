"""共享配置加载（Phase 4）。

训练侧与 Rust 后端共用同一个 config/backend.json：
  - Rust BackendConfig 读 ws/db/ollama/reward 等字段，忽略未知的 "training" 键；
  - 本模块只关心 db_path / persona_dir / model_name 以及 "training" 子节。
单一文件、单一事实源，避免两套配置漂移。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

# 仓库根目录（training/ 的上一级），用于把相对路径解析成绝对路径。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "backend.json")

# training 子节缺省值（与 config/backend.json 内的默认保持一致；文件缺字段时兜底）。
_TRAINING_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "base_model": "qwen2.5:1.5b-instruct",
    "hf_base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_dir": "data/adapters",
    "dataset_dir": "data/datasets",
    "bucketer": "lexical",
    "bucket_min_size": 2,
    "holdout_frac": 0.2,
    # RWR 主线（ADR-2）：样本权重 = clip(exp(advantage/rwr_beta), 0, rwr_w_max)
    "rwr_beta": 0.3,
    "rwr_w_max": 5.0,
    "min_samples": 20,
    # DPO 选做支路（近似构造，见 train_dpo.py 头注释）
    "min_pairs": 16,
    "pair_margin": 0.15,
    "pair_topk": 3,
    "dpo_beta": 0.1,
    "learning_rate": 5e-5,
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 1024,
    "proxy_min_accuracy": 0.5,
    "proxy_regress_epsilon": 0.02,
    "scheduler": {
        "nightly_start_hour": 2,
        "nightly_end_hour": 6,
        "min_new_samples": 50,
        "idle_required_secs": 600,
        "check_interval_secs": 600,
    },
}


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


@dataclass
class Config:
    """解析后的配置；所有路径已转成绝对路径。"""

    db_path: str
    persona_dir: str
    ollama_url: str
    training: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def adapter_dir(self) -> str:
        return _abs(self.training["adapter_dir"])

    @property
    def dataset_dir(self) -> str:
        return _abs(self.training["dataset_dir"])

    @property
    def scheduler(self) -> Dict[str, Any]:
        return self.training["scheduler"]


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | None = None) -> Config:
    path = path or os.environ.get("MC_CONFIG") or DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[config] 读取 {path} 失败（{e}），使用内置默认")
        raw = {}

    training = _deep_merge(_TRAINING_DEFAULTS, raw.get("training", {}))
    return Config(
        db_path=_abs(raw.get("db_path", "data/mood.db")),
        persona_dir=_abs(raw.get("persona_dir", "data/personas")),
        ollama_url=raw.get("ollama_url", "http://127.0.0.1:11434"),
        training=training,
        raw=raw,
    )
