"""夜间训练调度器（Phase 4，模块 E 的"触发条件：电脑闲置 + 凌晨 2-6 点"）。

三重门控，全部满足才触发一次 train_dpo：
    ① 时间窗：本地时间落在 [nightly_start_hour, nightly_end_hour)（支持跨午夜）
    ② 系统空闲：人机空闲 ≥ idle_required_secs（不抢用户在用时的算力）
    ③ 新样本：自上次训练以来新增 NORMAL 样本 ≥ min_new_samples（没有新偏好就别白训）
  外加"今夜已训过"短路，避免一晚反复触发。

运行方式：
    python -m training.scheduler --once     # 评估门控，满足就跑一次（推荐配 cron 用）
    python -m training.scheduler --daemon    # 常驻，每 check_interval_secs 评估一次
    python -m training.scheduler --now       # 跳过门控立即训练（手动/调试）

调度器只负责"何时触发"，真正的训练在子进程 `python -m training.train_dpo` 里跑，
进程隔离让重训练的显存/异常不会拖垮常驻调度器。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

from . import config as mc_config
from . import dataset as mc_dataset


# ----------------------------------------------------------------------------
# 门控①：时间窗
# ----------------------------------------------------------------------------
def in_nightly_window(now: datetime, start_h: int, end_h: int) -> bool:
    h = now.hour
    if start_h == end_h:
        return True  # 全天（不推荐，仅退化情形）
    if start_h < end_h:
        return start_h <= h < end_h
    return h >= start_h or h < end_h  # 跨午夜，如 22→6


# ----------------------------------------------------------------------------
# 门控②：系统空闲（尽力而为，跨平台；未知时不阻塞，仅告警）
# ----------------------------------------------------------------------------
def idle_seconds() -> Optional[float]:
    """返回人机空闲秒数；无法探测返回 None。"""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=5
            ).stdout
            for line in out.splitlines():
                if "HIDIdleTime" in line:
                    ns = int(line.rsplit("=", 1)[1].strip())
                    return ns / 1e9
        except Exception:
            return None
        return None
    if sys.platform.startswith("linux"):
        import shutil

        if shutil.which("xprintidle"):
            try:
                ms = int(subprocess.run(["xprintidle"], capture_output=True, text=True, timeout=5).stdout)
                return ms / 1000.0
            except Exception:
                return None
        return None  # 无 xprintidle / 纯 headless
    return None


# ----------------------------------------------------------------------------
# 门控③：新样本数 + 今夜已训短路
# ----------------------------------------------------------------------------
def _last_train_ts(conn) -> Optional[str]:
    row = conn.execute("SELECT ts FROM train_runs ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def new_sample_count(conn, since_ts: Optional[str]) -> int:
    """自 since_ts 之后新增的可训练 NORMAL 样本数（ts 为同格式 rfc3339，可字典序比较）。"""
    if since_ts is None:
        return conn.execute(
            "SELECT COUNT(*) FROM turns WHERE mode='NORMAL' AND excluded=0 AND reward IS NOT NULL"
        ).fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM turns WHERE mode='NORMAL' AND excluded=0 AND reward IS NOT NULL AND ts > ?",
        (since_ts,),
    ).fetchone()[0]


def trained_tonight(last_ts: Optional[str], now: datetime, start_h: int, end_h: int) -> bool:
    """本夜窗口内是否已训练过（避免一晚多次触发）。"""
    if last_ts is None:
        return False
    try:
        last = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return False
    # 粗判：上次训练距今不足 (窗口跨度+1) 小时，视为今夜已训
    span = (end_h - start_h) % 24 or 24
    return (now - last.replace(tzinfo=None)).total_seconds() < (span + 1) * 3600


# ----------------------------------------------------------------------------
# 决策 + 触发
# ----------------------------------------------------------------------------
def should_run(cfg: mc_config.Config, now: datetime) -> tuple[bool, str]:
    s = cfg.scheduler
    if not cfg.training.get("enabled", True):
        return False, "training.enabled=false"
    if not in_nightly_window(now, int(s["nightly_start_hour"]), int(s["nightly_end_hour"])):
        return False, f"不在夜间窗口 [{s['nightly_start_hour']},{s['nightly_end_hour']})"

    idle = idle_seconds()
    need_idle = float(s["idle_required_secs"])
    if idle is not None and idle < need_idle:
        return False, f"系统未空闲（{idle:.0f}s < {need_idle:.0f}s）"
    if idle is None:
        print("[sched] 无法探测空闲（headless？）→ 跳过空闲门控")

    conn = mc_dataset.connect(cfg.db_path)
    try:
        last_ts = _last_train_ts(conn)
        if trained_tonight(last_ts, now, int(s["nightly_start_hour"]), int(s["nightly_end_hour"])):
            return False, "今夜已训练过"
        new_n = new_sample_count(conn, last_ts)
    finally:
        conn.close()

    if new_n < int(s["min_new_samples"]):
        return False, f"新样本不足（{new_n} < {s['min_new_samples']}）"
    return True, f"满足全部门控（新样本={new_n}, 空闲={idle}）"


def trigger_training(config_path: Optional[str]) -> int:
    # 主线 = RWR（ADR-2）。DPO 是选做支路，需手动 `python -m training.train_dpo`。
    cmd = [sys.executable, "-m", "training.train"]
    if config_path:
        cmd += ["--config", config_path]
    print("[sched] 触发训练:", " ".join(cmd))
    return subprocess.run(cmd, cwd=mc_config.REPO_ROOT).returncode


def _main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 夜间训练调度器")
    ap.add_argument("--config", default=None)
    ap.add_argument("--once", action="store_true", help="评估门控一次，满足则训练（配 cron 用）")
    ap.add_argument("--daemon", action="store_true", help="常驻轮询")
    ap.add_argument("--now", action="store_true", help="跳过门控立即训练")
    args = ap.parse_args()

    cfg = mc_config.load(args.config)

    if args.now:
        return trigger_training(args.config)

    def tick() -> None:
        now = datetime.now()
        ok, why = should_run(cfg, now)
        print(f"[sched] {now.isoformat(timespec='seconds')} 决策: {'RUN' if ok else 'SKIP'} — {why}")
        if ok:
            trigger_training(args.config)

    if args.daemon:
        interval = float(cfg.scheduler["check_interval_secs"])
        print(f"[sched] 常驻模式，每 {interval:.0f}s 评估一次")
        while True:
            try:
                tick()
            except Exception as e:  # 调度器必须自愈，单次异常不退出
                print(f"[sched] 本轮异常（忽略）：{e}")
            time.sleep(interval)

    # 默认 = --once
    tick()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
