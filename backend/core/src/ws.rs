//! WS 服务（数据契约第 5 节）：监听 127.0.0.1，首条消息 token 校验。
//! 消息类型：auth / style_request / log_turn / crisis_event / persona_create(扩展) / error。
//! req_id 为契约的加法扩展：客户端带则原样回传，用于请求-响应关联。

use crate::config::BackendConfig;
use crate::db::{self, Db, TurnLog};
use crate::llm::StyleLlm;
use crate::persona::PersonaEngine;
use crate::reward;
use crate::style;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;

pub struct AppState {
    pub cfg: BackendConfig,
    pub db: Db,
    pub llm: Arc<dyn StyleLlm>,
    pub personas: PersonaEngine,
}

pub async fn serve(state: Arc<AppState>) -> Result<()> {
    let addr = format!("{}:{}", state.cfg.ws_host, state.cfg.ws_port);
    let listener = TcpListener::bind(&addr).await?;
    println!("[ws] listening on ws://{addr}");
    loop {
        let (stream, peer) = listener.accept().await?;
        let st = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(st, stream).await {
                eprintln!("[ws] 连接 {peer} 结束：{e}");
            }
        });
    }
}

async fn handle_conn(state: Arc<AppState>, stream: TcpStream) -> Result<()> {
    let ws = tokio_tungstenite::accept_async(stream).await?;
    let (mut tx, mut rx) = ws.split();

    // ---- 首条消息必须是 auth（简单 token，防本地误连）----
    let authed = match rx.next().await {
        Some(Ok(Message::Text(s))) => {
            let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
            v["type"] == "auth" && v["token"] == state.cfg.ws_token.as_str()
        }
        _ => false,
    };
    if !authed {
        let _ = tx
            .send(Message::Text(
                json!({"type":"error","code":"auth_failed","message":"token 校验失败"}).to_string(),
            ))
            .await;
        return Ok(());
    }
    tx.send(Message::Text(json!({"type":"auth_ok"}).to_string())).await?;

    while let Some(msg) = rx.next().await {
        let msg = msg?;
        let Message::Text(text) = msg else {
            if matches!(msg, Message::Close(_)) { break; }
            continue;
        };
        let v: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                tx.send(Message::Text(err_msg(None, "bad_json", "消息不是合法 JSON"))).await?;
                continue;
            }
        };
        let req_id = v["req_id"].as_str().map(|s| s.to_string());
        let reply = dispatch(&state, &v, req_id.clone()).await;
        match reply {
            Ok(Some(r)) => tx.send(Message::Text(r)).await?,
            Ok(None) => {}
            Err(e) => tx.send(Message::Text(err_msg(req_id, "internal", &e.to_string()))).await?,
        }
    }
    Ok(())
}

fn err_msg(req_id: Option<String>, code: &str, message: &str) -> String {
    json!({"type":"error","req_id":req_id,"code":code,"message":message}).to_string()
}

async fn dispatch(state: &Arc<AppState>, v: &Value, req_id: Option<String>) -> Result<Option<String>> {
    match v["type"].as_str().unwrap_or("") {
        // ---- 模块 B：style_request → style_response ----
        "style_request" => {
            let session_id = v["session_id"].as_str().unwrap_or("").to_string();
            let persona_id = v["persona_id"].as_str().unwrap_or("default");
            let user_input = v["user_input"].as_str().unwrap_or("");
            let persona = state.personas.load_or_default(persona_id);
            let style_payload = style::generate_style(
                state.llm.as_ref(),
                &persona,
                user_input,
                state.cfg.style_temperature,
                state.cfg.style_max_tokens,
            )
            .await?;
            Ok(Some(
                json!({
                    "type": "style_response",
                    "req_id": req_id,
                    "session_id": session_id,
                    "style_payload": style_payload,
                    "model_version": db::current_model_version(&state.db)
                })
                .to_string(),
            ))
        }

        // ---- 模块 C：log_turn → 奖励合成（第 6 节）→ 落库 ----
        "log_turn" => {
            let t: TurnLog = serde_json::from_value(v.clone())?;
            let mode = t.mode.clone().unwrap_or_else(|| "NORMAL".into());
            // 数据隔离（ADR-4）：CRITICAL 不计算 reward、excluded=1（insert_turn 内强制）
            let (reward_val, components) = if mode == "CRITICAL" {
                (None, None)
            } else {
                let n = db::behavior_sample_count(&state.db, &t.persona_id).unwrap_or(0);
                let (r, c) = reward::compute_reward(&state.cfg.reward, &t, n);
                (Some(r), Some(c))
            };
            let id = db::insert_turn(&state.db, &t, reward_val, components)?;
            Ok(Some(
                json!({"type":"log_ack","req_id":req_id,"turn_id":id,"reward":reward_val}).to_string(),
            ))
        }

        // ---- 模块 D：危机事件记录 ----
        "crisis_event" => {
            let session_id = v["session_id"].as_str().unwrap_or("");
            let source = v["source"].as_str().unwrap_or("unknown");
            db::insert_crisis_event(&state.db, session_id, source)?;
            Ok(Some(json!({"type":"crisis_ack","req_id":req_id}).to_string()))
        }

        // ---- 模块 A：人格扩写（Tauri UI / CLI 调用；契约加法扩展）----
        "persona_create" => {
            let one_liner = v["one_liner"].as_str().unwrap_or("").to_string();
            if one_liner.is_empty() {
                return Ok(Some(err_msg(req_id, "bad_request", "one_liner 不能为空")));
            }
            let (cfg, used_fallback) = state
                .personas
                .expand(state.llm.as_ref(), &one_liner, state.cfg.persona_temperature)
                .await;
            Ok(Some(
                json!({
                    "type": "persona_response",
                    "req_id": req_id,
                    "persona_id": cfg.persona_id,
                    "used_fallback": used_fallback,
                    "config": serde_json::to_value(&cfg)?
                })
                .to_string(),
            ))
        }

        other => Ok(Some(err_msg(req_id, "unknown_type", &format!("未知消息类型 {other}")))),
    }
}
