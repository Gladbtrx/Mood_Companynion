"""train_runs 表读写助手（RWR 主线 train.py 与 DPO 选做支路 train_dpo.py 共用）。

与 backend/core/src/db.rs 的语义严格对齐：
  - current/prev_effective_version：最近一条 rolled_back=0 的 model_version，默认 0（基座）；
  - 回滚 = 写入 rolled_back=1 的行，Rust 端 current_model_version 自动忽略它。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional


def prev_effective_version(conn: sqlite3.Connection) -> int:
    """当前生效版本（与 db.rs::current_model_version 完全一致）。"""
    row = conn.execute(
        "SELECT model_version FROM train_runs WHERE rolled_back = 0 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else 0


def best_proxy_metric(conn: sqlite3.Connection) -> Optional[float]:
    row = conn.execute(
        "SELECT proxy_metric FROM train_runs WHERE rolled_back = 0 AND proxy_metric IS NOT NULL "
        "ORDER BY proxy_metric DESC LIMIT 1"
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def record_train_run(
    conn: sqlite3.Connection,
    *,
    model_version: int,
    n_samples: int,
    proxy_metric: Optional[float],
    rolled_back: bool,
) -> int:
    cur = conn.execute(
        "INSERT INTO train_runs (ts, model_version, n_samples, proxy_metric, rolled_back) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            model_version,
            n_samples,
            proxy_metric,
            1 if rolled_back else 0,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def decide_rollback(
    conn: sqlite3.Connection,
    proxy: Optional[float],
    *,
    min_metric: float,
    regress_epsilon: float,
) -> tuple[bool, str]:
    """回滚护栏（模块 E"指标不升则自动回滚"）：
    proxy 为 None（留出集为空）→ 保守不回滚但指标记 null；
    低于绝对阈值，或较历史最佳退化超 epsilon → 回滚。"""
    if proxy is None:
        return False, "留出集为空，无法评估（proxy=null，保守不回滚）"
    if proxy < min_metric:
        return True, f"低于绝对阈值 {min_metric}"
    best = best_proxy_metric(conn)
    if best is not None and proxy < best - regress_epsilon:
        return True, f"较历史最佳 {best:.4f} 退化 >{regress_epsilon}"
    return False, ""
