#!/usr/bin/env bash
# 注册 Phase 4 夜间训练调度（"凌晨触发"）。
#
# 策略：cron 每晚 02:05 拉起 `scheduler.py --once`，由调度器内部再做
#       时间窗 / 空闲 / 新样本三重门控 —— cron 只负责"把调度器叫醒"，
#       真正要不要训练由 should_run() 决定。这样窗口/阈值改配置即可，无需改 crontab。
#
# 用法：  bash scripts/install-cron.sh           # 安装
#         bash scripts/install-cron.sh --remove  # 卸载
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${MC_PYTHON:-python3}"
TAG="# mood-companion-nightly-dpo"
LOG="$REPO_ROOT/data/train.log"
LINE="5 2 * * * cd $REPO_ROOT && $PYTHON -m training.scheduler --once >> $LOG 2>&1 $TAG"

current="$(crontab -l 2>/dev/null || true)"
cleaned="$(printf '%s\n' "$current" | grep -vF "$TAG" || true)"

if [[ "${1:-}" == "--remove" ]]; then
  printf '%s\n' "$cleaned" | crontab -
  echo "[cron] 已移除夜间训练任务"
  exit 0
fi

{ printf '%s\n' "$cleaned"; printf '%s\n' "$LINE"; } | crontab -
echo "[cron] 已安装夜间训练任务（每日 02:05 唤醒调度器）："
echo "       $LINE"
echo "[cron] 日志：$LOG"
echo "[cron] macOS 提示：需在 系统设置→隐私与安全性→完全磁盘访问 给 cron 授权，否则读不到 DB。"
