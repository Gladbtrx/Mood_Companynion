//! 本地推理后端抽象（ADR-5：默认 Ollama 主路径；trait 化以便切换到
//! transformers 直载 LoRA 的免转换备选路径 —— 届时实现同一 trait 即可）。

use anyhow::{Context, Result};
use serde_json::json;

#[async_trait::async_trait]
pub trait StyleLlm: Send + Sync {
    async fn generate(&self, prompt: &str, temperature: f64, max_tokens: u32) -> Result<String>;
}

pub struct OllamaClient {
    pub base_url: String,
    pub model: String,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub async fn ping(&self) -> bool {
        self.http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[async_trait::async_trait]
impl StyleLlm for OllamaClient {
    async fn generate(&self, prompt: &str, temperature: f64, max_tokens: u32) -> Result<String> {
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": { "temperature": temperature, "num_predict": max_tokens }
        });
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await
            .context("ollama 不可达")?;
        let v: serde_json::Value = resp.json().await.context("ollama 响应非 JSON")?;
        v.get("response")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .context("ollama 响应缺少 response 字段")
    }
}
