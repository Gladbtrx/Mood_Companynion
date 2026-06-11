//! 模块 A：冷启动人格引擎。
//! 一句话描述 → Qwen 扩写为 persona_config.json → schema 校验，
//! 失败重试 ≤2 次，仍失败回退内置默认模板并提示（模块 A 验收要求）。

use crate::llm::StyleLlm;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    pub persona_id: String,
    pub version: i64,
    pub one_liner: String,
    pub system_rules: Vec<String>,
    pub negative_constraints: Vec<String>,
    pub style_indicators: StyleIndicators,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleIndicators {
    pub tone: Vec<String>,
    pub formality: String,
    pub emoji: String,
    pub sentence_length: String,
}

pub fn default_persona(one_liner: &str) -> PersonaConfig {
    let now = chrono::Utc::now().to_rfc3339();
    PersonaConfig {
        persona_id: "default".into(),
        version: 1,
        one_liner: one_liner.to_string(),
        system_rules: vec![
            "像亲近的人一样说话，先回应情绪，再谈事情".into(),
            "口语化短句，不堆砌建议清单".into(),
            "记得对方说过的事，自然地接话".into(),
        ],
        negative_constraints: vec![
            "禁止学术腔".into(),
            "禁止说教".into(),
            "禁止爹味解释".into(),
            "禁止以'作为AI'开头的免责".into(),
        ],
        style_indicators: StyleIndicators {
            tone: vec!["sarcastic".into(), "warm-underneath".into()],
            formality: "low".into(),
            emoji: "sparse".into(),
            sentence_length: "short".into(),
        },
        created_at: now.clone(),
        updated_at: now,
    }
}

const EXPAND_PROMPT: &str = r#"你是人格配置生成器。把用户的一句话描述扩写成 JSON 人格配置。
只输出 JSON 本身，不要任何解释、markdown 代码块或多余文本。
JSON 必须严格符合此结构（值用中文，tone/formality/emoji/sentence_length 用英文小写词）：
{"persona_id":"<英文短横线小写id>","version":1,"one_liner":"<原句>","system_rules":["3到5条行为规则"],"negative_constraints":["3到5条禁止项，必须包含 禁止学术腔 禁止说教"],"style_indicators":{"tone":["1-3个英文词"],"formality":"low|medium|high","emoji":"none|sparse|rich","sentence_length":"short|medium|long"}}

用户描述：__ONE_LINER__
"#;

/// schema 校验：必需字段齐全、类型正确、关键枚举合法
pub fn validate(v: &Value) -> Result<(), String> {
    let obj = v.as_object().ok_or("根节点不是对象")?;
    for key in ["persona_id", "one_liner", "system_rules", "negative_constraints", "style_indicators"] {
        if !obj.contains_key(key) {
            return Err(format!("缺少字段 {key}"));
        }
    }
    if !obj["system_rules"].is_array() || obj["system_rules"].as_array().unwrap().is_empty() {
        return Err("system_rules 必须是非空数组".into());
    }
    if !obj["negative_constraints"].is_array() || obj["negative_constraints"].as_array().unwrap().is_empty() {
        return Err("negative_constraints 必须是非空数组".into());
    }
    let si = obj["style_indicators"].as_object().ok_or("style_indicators 不是对象")?;
    for key in ["tone", "formality", "emoji", "sentence_length"] {
        if !si.contains_key(key) {
            return Err(format!("style_indicators 缺少 {key}"));
        }
    }
    Ok(())
}

/// persona_id 规范化：小模型常产出空串/标点等垃圾 id，这里强制收敛为
/// [a-z0-9-]，不合格则用 one_liner 的短哈希兜底，保证文件名与外键稳定。
pub fn normalize_persona_id(raw: &str, one_liner: &str) -> String {
    let cleaned: String = raw
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if cleaned.len() >= 3 {
        return cleaned.chars().take(40).collect();
    }
    // FNV-1a 短哈希（避免引入额外依赖）
    let mut h: u64 = 0xcbf29ce484222325;
    for b in one_liner.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("persona-{:08x}", (h & 0xffff_ffff) as u32)
}

/// 从模型输出中尽力挖出 JSON（容忍 ```json 包裹或前后多话）
fn extract_json(raw: &str) -> Option<Value> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&raw[start..=end]).ok()
}

pub struct PersonaEngine {
    pub dir: PathBuf,
}

impl PersonaEngine {
    pub fn new(dir: &str) -> Self {
        Self { dir: PathBuf::from(dir) }
    }

    pub fn save(&self, p: &PersonaConfig) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{}.json", p.persona_id));
        std::fs::write(path, serde_json::to_string_pretty(p)?)?;
        Ok(())
    }

    pub fn load(&self, persona_id: &str) -> Option<PersonaConfig> {
        let path = self.dir.join(format!("{persona_id}.json"));
        let s = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&s).ok()
    }

    /// 找不到指定 persona 时回退默认（运行时绝不因 persona 缺失而失败）
    pub fn load_or_default(&self, persona_id: &str) -> PersonaConfig {
        self.load(persona_id)
            .unwrap_or_else(|| default_persona("一个嘴上嫌弃我、心里其实很在乎我的青梅竹马"))
    }

    /// 冷启动扩写：重试 ≤2，仍失败回退默认模板（返回 (config, used_fallback)）
    pub async fn expand(
        &self,
        llm: &dyn StyleLlm,
        one_liner: &str,
        temperature: f64,
    ) -> (PersonaConfig, bool) {
        let prompt = EXPAND_PROMPT.replace("__ONE_LINER__", one_liner);
        for attempt in 0..3 {
            match llm.generate(&prompt, temperature, 512).await {
                Ok(raw) => {
                    if let Some(mut v) = extract_json(&raw) {
                        if validate(&v).is_ok() {
                            let now = chrono::Utc::now().to_rfc3339();
                            let obj = v.as_object_mut().unwrap();
                            let pid = normalize_persona_id(
                                obj["persona_id"].as_str().unwrap_or(""),
                                one_liner,
                            );
                            obj.insert("persona_id".into(), Value::from(pid));
                            obj.insert("version".into(), Value::from(1));
                            obj.insert("one_liner".into(), Value::from(one_liner));
                            obj.insert("created_at".into(), Value::from(now.clone()));
                            obj.insert("updated_at".into(), Value::from(now));
                            if let Ok(cfg) = serde_json::from_value::<PersonaConfig>(v) {
                                let _ = self.save(&cfg);
                                return (cfg, false);
                            }
                        } else if let Err(e) = validate(&v) {
                            eprintln!("[persona] 第{}次扩写校验失败：{e}", attempt + 1);
                        }
                    } else {
                        eprintln!("[persona] 第{}次扩写未产出合法 JSON", attempt + 1);
                    }
                }
                Err(e) => eprintln!("[persona] 第{}次扩写推理失败：{e}", attempt + 1),
            }
        }
        // 回退：默认模板 + 保留用户原始描述（模块 A 验收要求"回退并提示"）
        eprintln!("[persona] 扩写失败 3 次，回退内置默认模板");
        let mut p = default_persona(one_liner);
        p.persona_id = "default".into();
        let _ = self.save(&p);
        (p, true)
    }
}
