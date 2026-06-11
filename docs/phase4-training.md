# 阶段 4 设计取舍：从 bandit 日志到本地夜间训练

本文档解释夜间训练管线（`training/`）的关键设计决策，以及它如何接进已有的数据契约
（`turns` / `train_runs` 表、`reward.rs`、`current_model_version`）。

## 1. 数据形态决定算法：RWR 是主线（ADR-2）

日志是 contextual bandit 形态——每轮只有 `(x=user_input, a=style_payload, r=reward)`：
"对这句话，模型采了这个风格，得到这个分"。**没有同一 x 下的两个候选风格**，
所以合法的 DPO 三元组 `(x, y_w, y_l)` 根本构造不出来。

RWR（Reward-Weighted Regression）正面消费这种数据：对每条样本做监督微调，
但 loss 乘以与奖励挂钩的权重

```
w = clip( exp(advantage / β), 0, w_max )
```

- 高 advantage 的风格 → 权重指数放大 → 生成概率上升；
- 低/负 advantage 的风格 → 权重趋零（截断后直接丢弃）→ 不被强化；
- `w_max` 截断防止个别异常高分样本主导整轮训练。

实现要点（`train.py::WeightedTrainer`）：逐样本算 completion 部分的平均 token loss
（prompt 部分 labels=-100），乘以样本权重后求均值。

## 2. 语义分桶 + advantage：去掉话题混杂

裸 `reward` 不能直接比较：emo 话题下用户普遍更愿意复制回复，高 reward 可能来自
话题而非风格。于是：

```
bucket(x)  = 语义桶（lexical 关键词桶，可选 embedding+KMeans）
advantage  = reward − mean_reward(bucket)
```

`advantage` 正是 `reward.rs` 头注释里写明"由夜间训练管线统一计算并回写
`turns.advantage`"的那一列——运行时刻意不算它，因为它需要看到整桶样本。

## 3. DPO 仅作选做支路，且声明近似有偏

`train_dpo.py` 按 ADR-2 的许可保留为对比实验支路：桶内高 advantage 当 chosen、
低 advantage 当 rejected。两个已声明的偏差源：

1. chosen/rejected 来自**不同的用户消息**，共享 prompt 取 chosen 轮的 context，
   依赖"同桶语境相近"的假设；
2. 学到的可能是"桶内话题差异"而非纯风格偏好。

报告中引用该支路结果必须注明以上局限。日常训练一律走 `train.py`。

## 4. 被微调的是"风格控制器"，不是聊天模型

优化对象是本地 SLM（Qwen2.5-1.5B-Instruct）作为 `<style>` 生成器的行为。训练复刻的
prompt 必须与运行时 `style.rs::STYLE_PROMPT` **逐字一致**，否则梯度作用在错误的输入
分布上（train/serve skew）。`training/prompts.py` 是其镜像，
`python3 -m training.dataset --selfcheck` 读 Rust 源码常量做逐字比对。

## 5. 代理指标 + 回滚：宁可不进步，不可退步

本地训练没有人工标注集，护栏是留出集代理指标（模块 E："平均 advantage /
风格分类器命中率"）：

- mock 路径：留出集加权平均 advantage 过 sigmoid（确定性、可复现）；
- 真训练（TODO-GPU）：新旧权重在留出 prompt 上生成风格，比较落入高 advantage
  风格簇的比例。

回滚条件（`runlog.decide_rollback`，RWR 与 DPO 支路共用）：低于绝对阈值
`proxy_min_accuracy`，或较历史最佳退化超 `proxy_regress_epsilon` → 写入
`rolled_back=1`。`db.rs::current_model_version` 只认最近一条 `rolled_back=0`
的版本，回滚版本被自动忽略，次日继续用旧权重——防止某夜噪声数据把人设训崩。

## 6. 与 Rust 后端的接缝

| 接缝 | 谁写 | 谁读 |
|---|---|---|
| `turns.advantage` | 训练管线回写 | 训练管线（加权/配对） |
| `turns.excluded` | 后端 `insert_turn`（CRITICAL→1） | 训练管线（只取 excluded=0） |
| `train_runs` | `runlog.record_train_run` | 后端 `current_model_version` 选权重 |
| `config.training.*` | 人工 | 训练管线（Rust 忽略此未知键） |
| `model_name` → `mood-companion:vN` | `export_ollama` 提示人工切换 | 后端 `style_request` 加载 |

训练侧用 Python（贴近 HF/peft/bitsandbytes 栈，ADR-5）；与 Rust 后端零进程耦合，
仅通过共享 SQLite 与配置文件协作。

## 7. 占位边界（当前仓库状态，诚实标注）

- **真实可跑**：加权样本构造、advantage 回写、RWR 权重公式、留出切分、代理指标、
  回滚护栏、`train_runs` 记录、调度门控、DPO 支路配对——均已在合成数据上端到端验证
  （CRITICAL 排除、权重公式逐样本核对、版本递增、回滚生效）。
- **骨架（# MOCK）**：QLoRA 反传需 GPU；GGUF/Ollama 导出需 llama.cpp + ollama。
  缺则自动降级（模拟训练 / 打印导出计划），装好真实栈后自动切换，调用方不变。
