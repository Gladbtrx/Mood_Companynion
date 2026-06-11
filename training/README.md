# 阶段 4 — 夜间本地 RWR 训练管线（模块 E / ADR-2）

把白天积累的"云端自评分 + 隐式行为"日志，在凌晨离线训练成**次日的人设进化**。
个性化策略、奖励信号与训练过程 100% 留在本机（注意：这不等于对话对云端不可见，见顶层 README 隐私口径）。

## 主线是 RWR，不是 DPO（ADR-2）

运行时每轮只发送一个 style、云端只回一个结果——**拿不到同一 context 下的
chosen/rejected 反事实对照，凑不出合法 DPO 数据对**。因此：

- **主线 `train.py`：RWR（奖励加权回归）** —— 把 `(context, style, reward)` 三元组
  做加权 QLoRA SFT，样本权重 `w = clip(exp(advantage/β), 0, w_max)`，提高高奖励
  风格的生成概率。免成对数据、与标量行为奖励天然契合。
- **选做支路 `train_dpo.py`：近似 DPO** —— 语义分桶后桶内高/低 advantage 配对。
  **近似且有偏**（chosen/rejected 来自不同消息），仅供对比实验，详见其文件头声明。

## 管线结构

```
scheduler.py（凌晨窗口 ∧ 系统空闲 ∧ 新样本数 三重门控）
   └─▶ train.py（RWR 主线）
         ├─ dataset.py    取 NORMAL/excluded=0 → 语义分桶 → advantage 回写 turns.advantage
         ├─ 加权样本      w = clip(exp(advantage/β), 0, w_max)，权重≈0 的强负样本丢弃
         ├─ QLoRA SFT     transformers+peft+bitsandbytes 4-bit；无 GPU / --mock 时模拟
         ├─ 代理指标      留出集加权平均 advantage → runlog.decide_rollback 不升则回滚
         ├─ train_runs    版本/指标/回滚记录（Rust current_model_version 据此选权重）
         └─ export_ollama LoRA → merge → GGUF → `ollama create mood-companion:vN` 热更新
```

数据隔离铁律（ADR-4）：`CRITICAL` 轮落库即 `excluded=1`，取数阶段天然排除，
**危机对话绝不进训练集**。

## 快速开始（无需 GPU，跑通闭环演示）

```bash
cd mood-companion

# 0. 校验训练侧 prompt 与 Rust style.rs 逐字一致（防 train/serve skew）
python3 -m training.dataset --selfcheck

# 1. 主线 RWR：mock 模式跑完整回环（数据→训练→评估→回滚判定→train_runs→导出计划）
python3 -m training.train --mock --force

# 2. 评估调度门控（不满足时打印 SKIP 原因）
python3 -m training.scheduler --once

# 3.（选做）近似 DPO 支路
python3 -m training.train_dpo --dry-run --force
```

`--mock` 强制模拟训练且不导出；`--force` 忽略 `enabled` 与最小样本门槛。

## 真训练（NVIDIA GPU）

```bash
pip install -r training/requirements.txt
python3 -m training.train        # 检测到 GPU+HF 栈自动切真 QLoRA 加权 SFT
```

导出回环需 [llama.cpp](https://github.com/ggerganov/llama.cpp)
（`convert_hf_to_gguf.py` / `llama-quantize`）与 [ollama](https://ollama.com)；
缺工具时 `export_ollama.py` 自动降级为打印导出计划，不影响训练记录。

## 安装夜间定时（凌晨触发）

```bash
bash scripts/install-cron.sh           # 每日 02:05 唤醒调度器，门控由 should_run() 决定
bash scripts/install-cron.sh --remove
```

或常驻：`python3 -m training.scheduler --daemon`。

## 调参（config/backend.json 的 training 子节）

| 键 | 作用 |
|---|---|
| `rwr_beta` / `rwr_w_max` | RWR 权重 `clip(exp(adv/β),0,w_max)`：β 越小对高 advantage 越激进 |
| `min_samples` | 加权样本不足则跳过本轮（RWR 主线门槛） |
| `bucketer` | `lexical`（默认零依赖）/ `embedding`（句向量+KMeans，桶更细） |
| `proxy_min_accuracy` / `proxy_regress_epsilon` | 回滚护栏：绝对阈值 + 相对退化容忍 |
| `lora_r` / `learning_rate` / `epochs` / `max_seq_len` | QLoRA 超参 |
| `min_pairs` / `pair_margin` / `pair_topk` | 仅 DPO 选做支路使用 |
| `scheduler.*` | 凌晨窗口、空闲秒数、最小新样本、轮询间隔 |

设计取舍详见 [`docs/phase4-training.md`](../docs/phase4-training.md)。
