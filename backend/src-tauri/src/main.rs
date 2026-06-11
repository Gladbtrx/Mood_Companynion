// Tauri 壳：启动时在后台线程拉起 headless core 的 WS 服务，
// WebView UI（../ui）与浏览器扩展走同一个 ws://127.0.0.1:8765。
// ⚠️ 本壳未在无 GUI 的开发机上编译验证过（缺 webkit2gtk），
//    核心逻辑均在 core 的 headless 路径验证，壳只做窗口与进程托管。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use mood_backend_core::{config::BackendConfig, db, llm::OllamaClient, persona::PersonaEngine, ws};
use std::sync::Arc;

fn spawn_backend() {
    std::thread::spawn(|| {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let cfg_path =
                std::env::var("MC_CONFIG").unwrap_or_else(|_| "config/backend.json".into());
            let cfg = BackendConfig::load(&cfg_path);
            if let Err(e) = cfg.ensure_dirs() {
                eprintln!("[tauri-shell] 目录初始化失败：{e}");
                return;
            }
            let database = match db::open(&cfg.db_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[tauri-shell] 数据库打开失败：{e}");
                    return;
                }
            };
            let ollama = OllamaClient::new(&cfg.ollama_url, &cfg.model_name);
            if !ollama.ping().await {
                eprintln!("[tauri-shell] 警告：Ollama 不可达，style 请求将降级");
            }
            let state = Arc::new(ws::AppState {
                personas: PersonaEngine::new(&cfg.persona_dir),
                db: database,
                llm: Arc::new(ollama),
                cfg,
            });
            if let Err(e) = ws::serve(state).await {
                eprintln!("[tauri-shell] WS 服务退出：{e}");
            }
        });
    });
}

fn main() {
    spawn_backend();
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("Tauri 启动失败");
}
