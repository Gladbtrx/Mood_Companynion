"""风格控制器提示词的 Python 镜像（Phase 4 训练侧）。

⚠️ 单一事实源同步要求：
    本文件的 STYLE_PROMPT 必须与 `backend/core/src/style.rs::STYLE_PROMPT`
    逐字一致。DPO 微调的本质是「让 SLM 在**它当初看到的同一段 prompt** 下，
    把高奖励的 <style> 排在低奖励的 <style> 之前」。如果训练时重建的 prompt
    与运行时（style.rs）不一致，偏好梯度就作用在错误的输入分布上，训练→推理
    之间会出现 train/serve skew，白练。

    改动二者任一处时，请同时改另一处，并跑 `python -m training.dataset --selfcheck`
    （会断言占位符集合一致）。

为什么把人设拼进 prompt：style.rs 在生成时已把 persona 字段填入，模型实际看到的
是「人设 + 用户这句话」。训练复刻同样的拼接，才能让 chosen/rejected 的对比落在
真实条件分布上。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

# 与 style.rs 逐字一致；占位符同样用 __XXX__ 包裹，便于 selfcheck 比对。
STYLE_PROMPT = """你是"风格控制器"。根据人格设定和用户这句话，生成一行简短的风格指令，
用来指导另一个 AI 如何回复用户。只输出风格指令本身，不要回复用户，不要解释。

要求：
- 一行中文（禁止夹杂英文单词），不超过 60 字；
- 是"如何说话"的指令，不是对用户说的话；
- 描述语气、称呼方式、句长、该突出的情绪反应；
- 必须遵守人格的禁止项。

示例（仅示范格式）：
用户这句话：老板又骂我了，我是不是真的很没用
风格指令：先狠狠帮腔骂老板一句，再别扭地夸对方，句子短促，结尾带一点笨拙的关心，禁止讲道理

人格设定：
- 一句话人设：__ONE_LINER__
- 行为规则：__RULES__
- 禁止项：__NEG__
- 语气关键词：__TONE__；正式度：__FORMALITY__；表情符号：__EMOJI__；句长：__SENTLEN__

用户这句话：__USER_INPUT__

风格指令："""

# style.rs 里 sanitize 后注入端强制 <style>…</style>，DB 存的就是带标签的串。
# 训练时 chosen/rejected 直接用 DB 原值，故无需再包标签。

# 默认人设（与 persona.rs::default_persona 对齐，persona 文件缺失时兜底）。
_DEFAULT_PERSONA: Dict[str, Any] = {
    "one_liner": "一个嘴上嫌弃我、心里其实很在乎我的青梅竹马",
    "system_rules": [
        "像亲近的人一样说话，先回应情绪，再谈事情",
        "口语化短句，不堆砌建议清单",
        "记得对方说过的事，自然地接话",
    ],
    "negative_constraints": ["禁止学术腔", "禁止说教", "禁止爹味解释", "禁止以'作为AI'开头的免责"],
    "style_indicators": {
        "tone": ["sarcastic", "warm-underneath"],
        "formality": "low",
        "emoji": "sparse",
        "sentence_length": "short",
    },
}


def load_persona(persona_dir: str, persona_id: str) -> Dict[str, Any]:
    """复刻 persona.rs::load_or_default：找不到就回退默认，绝不抛错。"""
    path = os.path.join(persona_dir, f"{persona_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_PERSONA)


def render_style_prompt(persona: Dict[str, Any], user_input: str) -> str:
    """逐字复刻 style.rs::generate_style 的占位符替换。"""
    si = persona.get("style_indicators", {}) or {}
    return (
        STYLE_PROMPT.replace("__ONE_LINER__", persona.get("one_liner", ""))
        .replace("__RULES__", "；".join(persona.get("system_rules", []) or []))
        .replace("__NEG__", "；".join(persona.get("negative_constraints", []) or []))
        .replace("__TONE__", ",".join(si.get("tone", []) or []))
        .replace("__FORMALITY__", si.get("formality", ""))
        .replace("__EMOJI__", si.get("emoji", ""))
        .replace("__SENTLEN__", si.get("sentence_length", ""))
        .replace("__USER_INPUT__", user_input or "")
    )


# selfcheck 用：占位符集合（与 style.rs 比对时用）
PLACEHOLDERS = {
    "__ONE_LINER__",
    "__RULES__",
    "__NEG__",
    "__TONE__",
    "__FORMALITY__",
    "__EMOJI__",
    "__SENTLEN__",
    "__USER_INPUT__",
}
