//! 端到端冒烟客户端：扮演浏览器扩展，按第 3 节数据流走一遍单轮闭环。
//! 用法：先启动 mood-backend，再 `cargo run --bin smoke`。
//! 退出码非 0 = 冒烟失败。

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

async fn expect_reply(
    rx: &mut (impl StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin),
    want_type: &str,
) -> Value {
    while let Some(Ok(Message::Text(s))) = rx.next().await {
        let v: Value = serde_json::from_str(&s).expect("回包不是 JSON");
        if v["type"] == want_type {
            return v;
        }
        if v["type"] == "error" {
            panic!("后端报错：{s}");
        }
    }
    panic!("连接中断，未等到 {want_type}");
}

#[tokio::main]
async fn main() {
    let url = std::env::args().nth(1).unwrap_or_else(|| "ws://127.0.0.1:8765".into());
    let token = std::env::var("MC_TOKEN").unwrap_or_else(|_| "mood-companion-dev-token".into());

    let (ws, _) = tokio_tungstenite::connect_async(&url).await.expect("无法连接后端");
    let (mut tx, mut rx) = ws.split();

    // 1. auth
    tx.send(Message::Text(json!({"type":"auth","token":token}).to_string())).await.unwrap();
    expect_reply(&mut rx, "auth_ok").await;
    println!("[smoke] auth ok");

    // 2. style_request（模块 B）
    tx.send(Message::Text(json!({
        "type":"style_request","req_id":"smoke-1",
        "session_id":"smoke-session","persona_id":"default",
        "user_input":"今天加班到十一点，好累，感觉一切都没有意义"
    }).to_string())).await.unwrap();
    let style = expect_reply(&mut rx, "style_response").await;
    let payload = style["style_payload"].as_str().unwrap_or("");
    assert!(payload.starts_with("<style>") && payload.ends_with("</style>"),
            "style_payload 必须是 <style>…</style>，得到：{payload}");
    println!("[smoke] style_response ok: {payload}");
    println!("[smoke] model_version = {}", style["model_version"]);

    // 3. log_turn（模块 C：正常轮，带行为信号 + 云端自评分）
    tx.send(Message::Text(json!({
        "type":"log_turn","req_id":"smoke-2",
        "session_id":"smoke-session","persona_id":"default",
        "user_input":"今天加班到十一点，好累","style_payload":payload,
        "raw_cloud_output":"<score>4</score><status>normal</status>哼，又熬夜。快去睡。",
        "clean_text":"哼，又熬夜。快去睡。",
        "cloud_score":4,"status":"normal","mode":"NORMAL",
        "regen_clicked":0,"copied":1,
        "followup_latency_ms":42000,"followup_len":25,
        "abandoned":0,"format_miss":0,"degraded":0,"finalize_reason":"next_response"
    }).to_string())).await.unwrap();
    let ack = expect_reply(&mut rx, "log_ack").await;
    let r = ack["reward"].as_f64().expect("NORMAL 轮必须有 reward");
    assert!((0.0..=1.0).contains(&r), "reward 必须在 [0,1]");
    println!("[smoke] log_turn(NORMAL) ok, turn_id={}, reward={r:.4}", ack["turn_id"]);

    // 4. CRITICAL 轮：必须无 reward（数据隔离，ADR-4）
    tx.send(Message::Text(json!({
        "type":"log_turn","req_id":"smoke-3",
        "session_id":"smoke-session","persona_id":"default",
        "user_input":"[危机测试样例]","mode":"CRITICAL","status":"crisis",
        "finalize_reason":"crisis_test"
    }).to_string())).await.unwrap();
    let ack2 = expect_reply(&mut rx, "log_ack").await;
    assert!(ack2["reward"].is_null(), "CRITICAL 轮不得计算 reward");
    println!("[smoke] log_turn(CRITICAL) ok, reward=null（已隔离）");

    // 5. crisis_event（模块 D）
    tx.send(Message::Text(json!({
        "type":"crisis_event","req_id":"smoke-4",
        "session_id":"smoke-session","source":"local_rule"
    }).to_string())).await.unwrap();
    expect_reply(&mut rx, "crisis_ack").await;
    println!("[smoke] crisis_event ok");

    // 6. 人格扩写（模块 A，走真实 Qwen，校验失败会自动回退默认）
    tx.send(Message::Text(json!({
        "type":"persona_create","req_id":"smoke-5",
        "one_liner":"一个嘴上嫌弃我、心里其实很在乎我的青梅竹马"
    }).to_string())).await.unwrap();
    let p = expect_reply(&mut rx, "persona_response").await;
    println!(
        "[smoke] persona_create ok: id={}, used_fallback={}",
        p["persona_id"], p["used_fallback"]
    );

    println!("\n[smoke] 全部通过 ✅");
}
