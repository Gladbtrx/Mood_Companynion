//! headless 后端入口：Tauri 壳（src-tauri）复用同一 AppState/serve，
//! 本机无 GUI 时可直接 `cargo run --bin mood-backend` 完整运行。

use mood_backend_core::{config::BackendConfig, db, llm::OllamaClient, persona::PersonaEngine, ws};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg_path = std::env::args().nth(1).unwrap_or_else(|| "config/backend.json".into());
    let cfg = BackendConfig::load(&cfg_path);
    cfg.ensure_dirs()?;

    let database = db::open(&cfg.db_path)?;
    let ollama = OllamaClient::new(&cfg.ollama_url, &cfg.model_name);
    if !ollama.ping().await {
        // 不退出：style_request 会失败 → 扩展走离线降级；Ollama 起来后自动恢复
        eprintln!(
            "[warn] Ollama 不可达（{}）。请先 `ollama serve` 并 `ollama pull {}`。",
            cfg.ollama_url, cfg.model_name
        );
    }

    let state = Arc::new(ws::AppState {
        personas: PersonaEngine::new(&cfg.persona_dir),
        db: database,
        llm: Arc::new(ollama),
        cfg,
    });
    ws::serve(state).await
}
