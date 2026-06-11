//! 风格控制器（ADR-1：本地 SLM 只产出 <style> 片段，不做主回复）。
//! 高温采样（T≈0.95）保证风格探索多样性 —— 这正是 RWR 训练需要的行为空间。

use crate::llm::StyleLlm;
use crate::persona::PersonaConfig;
use anyhow::Result;

const STYLE_PROMPT: &str = r#"你是"风格控制器"。根据人格设定和用户这句话，生成一行简短的风格指令，
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

风格指令："#;

/// 把模型输出清洗成单行 `<style>…</style>`（数据契约：注入端强制此格式）
pub fn sanitize_style(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    // 模型可能自己带了标签/引号/换行，全部拍平
    s = s
        .replace("<style>", "")
        .replace("</style>", "")
        .replace('\n', "；")
        .replace('"', "")
        .trim()
        .to_string();
    // 截断护栏：style 只是控制信号，过长会稀释云端注意力
    const MAX_CHARS: usize = 120;
    if s.chars().count() > MAX_CHARS {
        s = s.chars().take(MAX_CHARS).collect();
    }
    if s.is_empty() {
        // # MOCK 兜底：极端情况下给固定保底风格，绝不让扩展拿到空串
        s = "语气自然亲近，短句，先回应情绪再说事，禁止说教".into();
    }
    format!("<style>{s}</style>")
}

pub async fn generate_style(
    llm: &dyn StyleLlm,
    persona: &PersonaConfig,
    user_input: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String> {
    let prompt = STYLE_PROMPT
        .replace("__ONE_LINER__", &persona.one_liner)
        .replace("__RULES__", &persona.system_rules.join("；"))
        .replace("__NEG__", &persona.negative_constraints.join("；"))
        .replace("__TONE__", &persona.style_indicators.tone.join(","))
        .replace("__FORMALITY__", &persona.style_indicators.formality)
        .replace("__EMOJI__", &persona.style_indicators.emoji)
        .replace("__SENTLEN__", &persona.style_indicators.sentence_length)
        .replace("__USER_INPUT__", user_input);
    let raw = llm.generate(&prompt, temperature, max_tokens).await?;
    Ok(sanitize_style(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_wraps_and_flattens() {
        let s = sanitize_style("  <style>毒舌\n但暖</style> ");
        assert!(s.starts_with("<style>") && s.ends_with("</style>"));
        assert!(!s[7..s.len() - 8].contains('\n'));
    }

    #[test]
    fn sanitize_empty_falls_back() {
        let s = sanitize_style("   ");
        assert!(s.len() > "<style></style>".len());
    }

    #[test]
    fn sanitize_truncates() {
        let long = "很".repeat(500);
        let s = sanitize_style(&long);
        assert!(s.chars().count() <= 120 + "<style></style>".chars().count());
    }
}
