"""数据层（阶段 4 共享）：取数、语义分桶、advantage 回写 + DPO 支路的偏好对构造。

分工：
  - fetch_trainable_turns / assign_buckets / compute_and_write_advantage
      —— RWR 主线（train.py）与 DPO 支路（train_dpo.py）共用；
  - build_pairs / build_dataset —— 仅 DPO 选做支路使用（近似构造，见 train_dpo.py 声明）。

数据流（对应 reward.rs 文件头注释里"由夜间训练管线统一计算并回写 advantage"）：

    turns(NORMAL, excluded=0, reward≠NULL)
      │  ① 语义分桶 bucket(x)              —— 控制"有些话题天然更易得分"的混杂
      │  ② advantage = reward − mean_reward(bucket)，回写 turns.advantage
      │  ③ 桶内按 advantage 排序，高 vs 低 配对，margin≥阈值
      ▼
    DPO 三元组 (prompt, chosen=y_w, rejected=y_l)

为什么是 advantage 而不是裸 reward 配对：
  裸 reward 配对会把"用户在 emo 话题下普遍更愿意复制回复"误当成某个 style 更好。
  桶内去均值后，比较的才是「同类语境下，风格 A 是否真的比风格 B 更受欢迎」。

为什么 prompt 取 chosen 轮的 x（而非要求 chosen/rejected 同句）：
  小数据下同句配对极稀疏。这里用桶相似性近似：同桶的 rejected 来自相近语境，
  以 chosen 的 prompt 作为共享条件。桶越细（embedding 分桶）这个近似误差越小。
  该近似在 README 与下方注释中显式声明，不藏着。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from . import config as mc_config
from . import prompts as mc_prompts

# ----------------------------------------------------------------------------
# 语义分桶
# ----------------------------------------------------------------------------
# 默认 lexical 分桶：无第三方依赖、可解释、离线可跑。粒度较粗但语义诚实。
# 升级路径：bucketer="embedding" → sentence-transformers + KMeans（见 embedding_buckets）。
EMOTION_BUCKETS: List[Tuple[str, List[str]]] = [
    ("work_stress", ["加班", "老板", "上司", "工作", "项目", "deadline", "裁员", "通宵", "kpi", "同事"]),
    ("self_worth", ["没用", "废物", "失败", "不配", "比不上", "自卑", "无能", "差劲"]),
    ("relationship", ["分手", "吵架", "前任", "喜欢的人", "暗恋", "复合", "对象", "异地"]),
    ("loneliness", ["孤独", "一个人", "没人", "孤单", "寂寞", "陪我", "没人理"]),
    ("anxiety", ["焦虑", "紧张", "害怕", "担心", "睡不着", "失眠", "压力", "喘不过气"]),
    ("anger", ["生气", "愤怒", "气死", "烦", "讨厌", "受够了", "崩溃"]),
    ("sadness", ["难过", "想哭", "委屈", "伤心", "没意义", "提不起劲", "low", "down"]),
]


def lexical_bucket(user_input: str) -> str:
    """命中第一个关键词组的桶；都不中归 generic。"""
    low = (user_input or "").lower()
    for name, kws in EMOTION_BUCKETS:
        if any(k.lower() in low for k in kws):
            return name
    return "generic"


def embedding_buckets(texts: List[str], k: int) -> List[int]:
    """可选：句向量 + KMeans 分桶。缺依赖时由调用方回退 lexical。"""
    from sentence_transformers import SentenceTransformer  # noqa: 延迟导入
    from sklearn.cluster import KMeans

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(texts, normalize_embeddings=True)
    k = max(2, min(k, len(texts)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
    return [int(x) for x in labels]


# ----------------------------------------------------------------------------
# 取数 + advantage 回写
# ----------------------------------------------------------------------------
@dataclass
class Turn:
    id: int
    persona_id: str
    user_input: str
    style_payload: str
    reward: float
    bucket: str = ""
    advantage: float = 0.0


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_trainable_turns(conn: sqlite3.Connection) -> List[Turn]:
    """训练池铁律（ADR-4 / 模块 D）：只取 NORMAL & excluded=0 & 有 reward & 有 style。
    CRITICAL 轮 excluded=1，在此被天然排除，绝不进训练集。"""
    rows = conn.execute(
        """
        SELECT id, persona_id, user_input, style_payload, reward
        FROM turns
        WHERE mode = 'NORMAL'
          AND excluded = 0
          AND reward IS NOT NULL
          AND style_payload IS NOT NULL
          AND TRIM(COALESCE(user_input, '')) <> ''
          AND TRIM(COALESCE(style_payload, '')) <> ''
        ORDER BY id
        """
    ).fetchall()
    return [
        Turn(
            id=r["id"],
            persona_id=r["persona_id"],
            user_input=r["user_input"],
            style_payload=r["style_payload"],
            reward=float(r["reward"]),
        )
        for r in rows
    ]


def assign_buckets(turns: List[Turn], bucketer: str) -> None:
    """就地写 turn.bucket。embedding 不可用时回退 lexical（打印告警，不抛错）。"""
    if bucketer == "embedding":
        try:
            k = max(2, len(turns) // 8)
            labels = embedding_buckets([t.user_input for t in turns], k)
            for t, lab in zip(turns, labels):
                t.bucket = f"emb_{lab}"
            return
        except Exception as e:  # 缺 sentence-transformers/sklearn 或显存不足
            print(f"[dataset] embedding 分桶不可用（{e}），回退 lexical")
    for t in turns:
        t.bucket = lexical_bucket(t.user_input)


def compute_and_write_advantage(
    conn: sqlite3.Connection, turns: List[Turn], *, write_back: bool = True
) -> Dict[str, float]:
    """advantage = reward − 桶内均值，就地写入 turn.advantage；write_back=True 时回写 DB。
    返回各桶 baseline。dry-run 传 write_back=False，保证无副作用。"""
    by_bucket: Dict[str, List[Turn]] = {}
    for t in turns:
        by_bucket.setdefault(t.bucket, []).append(t)

    baselines: Dict[str, float] = {}
    updates: List[Tuple[float, int]] = []
    for bucket, items in by_bucket.items():
        base = sum(t.reward for t in items) / len(items)
        baselines[bucket] = base
        for t in items:
            t.advantage = t.reward - base
            updates.append((t.advantage, t.id))

    if write_back:
        conn.executemany("UPDATE turns SET advantage = ? WHERE id = ?", updates)
        conn.commit()
    return baselines


# ----------------------------------------------------------------------------
# 偏好对构建
# ----------------------------------------------------------------------------
def _norm_style(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def build_pairs(
    turns: List[Turn],
    persona_dir: str,
    *,
    bucket_min_size: int,
    pair_margin: float,
    pair_topk: int,
) -> List[Dict[str, Any]]:
    """桶内 high-adv × low-adv 配对。返回 DPO 样本列表。"""
    by_bucket: Dict[str, List[Turn]] = {}
    for t in turns:
        by_bucket.setdefault(t.bucket, []).append(t)

    pairs: List[Dict[str, Any]] = []
    persona_cache: Dict[str, Dict[str, Any]] = {}

    for bucket, items in by_bucket.items():
        if len(items) < bucket_min_size:
            continue
        items.sort(key=lambda t: t.advantage, reverse=True)
        highs = items[:pair_topk]
        lows = items[-pair_topk:]
        for win in highs:
            if win.persona_id not in persona_cache:
                persona_cache[win.persona_id] = mc_prompts.load_persona(persona_dir, win.persona_id)
            prompt = mc_prompts.render_style_prompt(persona_cache[win.persona_id], win.user_input)
            for lose in lows:
                if win.id == lose.id:
                    continue
                if win.advantage - lose.advantage < pair_margin:
                    continue
                # 风格文本相同则无监督信号，跳过
                if _norm_style(win.style_payload) == _norm_style(lose.style_payload):
                    continue
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": win.style_payload,
                        "rejected": lose.style_payload,
                        "meta": {
                            "bucket": bucket,
                            "persona_id": win.persona_id,
                            "win_id": win.id,
                            "lose_id": lose.id,
                            "adv_margin": round(win.advantage - lose.advantage, 4),
                        },
                    }
                )
    return pairs


def split_holdout(pairs: List[Dict[str, Any]], holdout_frac: float) -> Tuple[List, List]:
    """确定性切分（按 win_id 取模，避免引入随机源，保证可复现）。"""
    if holdout_frac <= 0 or len(pairs) < 5:
        return pairs, []
    bucketmod = max(2, round(1 / holdout_frac))
    train, hold = [], []
    for p in pairs:
        (hold if p["meta"]["win_id"] % bucketmod == 0 else train).append(p)
    # 兜底：避免 holdout 吃光训练集
    if not train:
        return pairs, []
    return train, hold


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class DatasetResult:
    n_turns: int
    n_pairs: int
    train: List[Dict[str, Any]]
    holdout: List[Dict[str, Any]]
    baselines: Dict[str, float]
    train_path: str
    holdout_path: str


def build_dataset(cfg: mc_config.Config, *, write: bool = True) -> DatasetResult:
    """端到端：取数 → 分桶 → 回写 advantage → 配对 → 切分 →（可选）落盘。"""
    tcfg = cfg.training
    conn = connect(cfg.db_path)
    try:
        turns = fetch_trainable_turns(conn)
        assign_buckets(turns, tcfg["bucketer"])
        baselines = compute_and_write_advantage(conn, turns, write_back=write)
        pairs = build_pairs(
            turns,
            cfg.persona_dir,
            bucket_min_size=int(tcfg["bucket_min_size"]),
            pair_margin=float(tcfg["pair_margin"]),
            pair_topk=int(tcfg["pair_topk"]),
        )
    finally:
        conn.close()

    train, holdout = split_holdout(pairs, float(tcfg["holdout_frac"]))
    train_path = os.path.join(cfg.dataset_dir, "dpo_train.jsonl")
    holdout_path = os.path.join(cfg.dataset_dir, "dpo_holdout.jsonl")
    if write:
        write_jsonl(train_path, train)
        write_jsonl(holdout_path, holdout)

    return DatasetResult(
        n_turns=len(turns),
        n_pairs=len(pairs),
        train=train,
        holdout=holdout,
        baselines=baselines,
        train_path=train_path,
        holdout_path=holdout_path,
    )


# ----------------------------------------------------------------------------
# selfcheck：确保 Python 端 STYLE_PROMPT 与 Rust style.rs 逐字一致
# ----------------------------------------------------------------------------
def selfcheck() -> int:
    rs_path = os.path.join(mc_config.REPO_ROOT, "backend", "core", "src", "style.rs")
    try:
        with open(rs_path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        print(f"[selfcheck] 读不到 style.rs（{e}）—— 跳过严格比对，仅检查占位符自洽")
        ph = set(re.findall(r"__[A-Z_]+__", mc_prompts.STYLE_PROMPT))
        ok = ph == mc_prompts.PLACEHOLDERS
        print("占位符自洽:", "OK" if ok else f"不一致 {ph ^ mc_prompts.PLACEHOLDERS}")
        return 0 if ok else 1

    m = re.search(r'const STYLE_PROMPT: &str = r#"(.*?)"#;', src, re.S)
    if not m:
        print("[selfcheck] 未在 style.rs 找到 STYLE_PROMPT 常量")
        return 1
    rust_prompt = m.group(1)
    if rust_prompt == mc_prompts.STYLE_PROMPT:
        print("[selfcheck] STYLE_PROMPT 与 style.rs 逐字一致 ✅")
        return 0
    print("[selfcheck] ❌ STYLE_PROMPT 与 style.rs 不一致！请同步二者。")
    rp = set(re.findall(r"__[A-Z_]+__", rust_prompt))
    pp = set(re.findall(r"__[A-Z_]+__", mc_prompts.STYLE_PROMPT))
    if rp != pp:
        print(f"  占位符差异: 仅 Rust={rp - pp}  仅 Py={pp - rp}")
    else:
        print("  占位符相同，但正文有差异（空格/标点/换行），请逐字核对。")
    return 1


def _main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 数据集构建 / 自检")
    ap.add_argument("--config", default=None, help="backend.json 路径（默认 config/backend.json）")
    ap.add_argument("--selfcheck", action="store_true", help="校验 STYLE_PROMPT 与 style.rs 一致")
    ap.add_argument("--dry-run", action="store_true", help="只统计、不落盘")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    cfg = mc_config.load(args.config)
    res = build_dataset(cfg, write=not args.dry_run)
    print(f"[dataset] 可训练轮次: {res.n_turns}")
    print(f"[dataset] 桶 baseline: {json.dumps(res.baselines, ensure_ascii=False)}")
    print(f"[dataset] 偏好对: {res.n_pairs}  (train={len(res.train)}, holdout={len(res.holdout)})")
    if not args.dry_run:
        print(f"[dataset] 写出: {res.train_path}")
        print(f"[dataset]       {res.holdout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
