//! 后端配置：所有权重/阈值集中于一个 JSON（第 6 节要求），便于调参与报告说明。

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendConfig {
    pub ws_host: String,
    pub ws_port: u16,
    /// 简单 token 校验：防其他本地进程误连，不是安全机制（数据契约第 5 节）
    pub ws_token: String,
    pub db_path: String,
    pub persona_dir: String,

    pub ollama_url: String,
    pub model_name: String,
    /// 风格生成用高随机性采样（第 3 节数据流第 3 步）
    pub style_temperature: f64,
    pub style_max_tokens: u32,
    /// 人格扩写用低温，保证 JSON 结构稳定（模块 A）
    pub persona_temperature: f64,

    pub reward: RewardConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardConfig {
    /// r_behavior = sigmoid(w1*copied + w2*engaged_followup - w3*regen - w4*abandoned)
    pub w_copied: f64,
    pub w_engaged: f64,
    pub w_regen: f64,
    pub w_abandoned: f64,
    /// engaged_followup 判定：追问长度下限 / 延迟上限
    pub engaged_min_len: i64,
    pub engaged_max_latency_ms: i64,
    /// alpha = N / (N + k)：冷启动混合（ADR-3）
    pub k_cold_start: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            w_copied: 1.0,
            w_engaged: 1.0,
            w_regen: 2.0,
            w_abandoned: 0.5,
            engaged_min_len: 8,
            engaged_max_latency_ms: 5 * 60 * 1000,
            k_cold_start: 20.0,
        }
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            ws_host: "127.0.0.1".into(),
            ws_port: 8765,
            ws_token: "mood-companion-dev-token".into(),
            db_path: "data/mood.db".into(),
            persona_dir: "data/personas".into(),
            ollama_url: "http://127.0.0.1:11434".into(),
            model_name: "qwen2.5:1.5b-instruct".into(),
            style_temperature: 0.95,
            style_max_tokens: 120,
            persona_temperature: 0.3,
            reward: RewardConfig::default(),
        }
    }
}

impl BackendConfig {
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => match serde_json::from_str(&s) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[config] {path} 解析失败（{e}），使用默认配置");
                    Self::default()
                }
            },
            Err(_) => {
                eprintln!("[config] 未找到 {path}，使用默认配置");
                Self::default()
            }
        }
    }

    pub fn ensure_dirs(&self) -> anyhow::Result<()> {
        if let Some(p) = Path::new(&self.db_path).parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::create_dir_all(&self.persona_dir)?;
        Ok(())
    }
}
