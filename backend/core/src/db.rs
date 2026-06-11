//! SQLite 存储（数据契约第 5 节 DDL + 两个加法扩展：
//! turns.degraded / turns.finalize_reason 列与 crisis_events 表，便于排查降级与危机触发）。

use anyhow::Result;
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};

pub type Db = Arc<Mutex<Connection>>;

pub const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL,
  persona_id TEXT NOT NULL,
  user_input TEXT,
  style_payload TEXT,
  raw_cloud_output TEXT,
  clean_text TEXT,
  cloud_score INTEGER,
  status TEXT,
  mode TEXT NOT NULL,
  regen_clicked INTEGER DEFAULT 0,
  copied INTEGER DEFAULT 0,
  followup_latency_ms INTEGER,
  followup_len INTEGER,
  abandoned INTEGER DEFAULT 0,
  format_miss INTEGER DEFAULT 0,
  reward REAL,
  reward_components TEXT,
  advantage REAL,
  excluded INTEGER DEFAULT 0,
  degraded INTEGER DEFAULT 0,
  finalize_reason TEXT
);
CREATE TABLE IF NOT EXISTS train_runs (
  id INTEGER PRIMARY KEY, ts TEXT, model_version INTEGER,
  n_samples INTEGER, proxy_metric REAL, rolled_back INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS crisis_events (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, session_id TEXT NOT NULL, source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_persona ON turns(persona_id);
"#;

pub fn open(path: &str) -> Result<Db> {
    let conn = Connection::open(path)?;
    conn.execute_batch(DDL)?;
    Ok(Arc::new(Mutex::new(conn)))
}

/// 扩展上报的一轮（log_turn 消息体）。serde(default) 容忍字段缺失 —— 第 7 节 #3/#4。
#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(default)]
pub struct TurnLog {
    pub ts: Option<String>,
    pub session_id: String,
    pub persona_id: String,
    pub user_input: Option<String>,
    pub style_payload: Option<String>,
    pub raw_cloud_output: Option<String>,
    pub clean_text: Option<String>,
    pub cloud_score: Option<i64>,
    pub status: Option<String>,
    pub mode: Option<String>,
    pub regen_clicked: i64,
    pub copied: i64,
    pub followup_latency_ms: Option<i64>,
    pub followup_len: Option<i64>,
    pub abandoned: i64,
    pub format_miss: i64,
    pub degraded: i64,
    pub finalize_reason: Option<String>,
}

pub fn insert_turn(
    db: &Db,
    t: &TurnLog,
    reward: Option<f64>,
    reward_components: Option<String>,
) -> Result<i64> {
    let mode = t.mode.clone().unwrap_or_else(|| "NORMAL".into());
    // 数据隔离铁律（ADR-4 / 模块 D）：CRITICAL 一律 excluded=1，绝不进训练集
    let excluded = if mode == "CRITICAL" { 1 } else { 0 };
    let conn = db.lock().unwrap();
    conn.execute(
        r#"INSERT INTO turns
           (ts, session_id, persona_id, user_input, style_payload, raw_cloud_output,
            clean_text, cloud_score, status, mode, regen_clicked, copied,
            followup_latency_ms, followup_len, abandoned, format_miss,
            reward, reward_components, advantage, excluded, degraded, finalize_reason)
           VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,NULL,?19,?20,?21)"#,
        params![
            t.ts.clone().unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
            t.session_id,
            t.persona_id,
            t.user_input,
            t.style_payload,
            t.raw_cloud_output,
            t.clean_text,
            t.cloud_score,
            t.status,
            mode,
            t.regen_clicked,
            t.copied,
            t.followup_latency_ms,
            t.followup_len,
            t.abandoned,
            t.format_miss,
            reward,
            reward_components,
            excluded,
            t.degraded,
            t.finalize_reason,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn insert_crisis_event(db: &Db, session_id: &str, source: &str) -> Result<()> {
    let conn = db.lock().unwrap();
    conn.execute(
        "INSERT INTO crisis_events (ts, session_id, source) VALUES (?1, ?2, ?3)",
        params![chrono::Utc::now().to_rfc3339(), session_id, source],
    )?;
    Ok(())
}

/// 该 persona 已积累的行为样本数 N（冷启动 alpha 用，第 6 节）
pub fn behavior_sample_count(db: &Db, persona_id: &str) -> Result<i64> {
    let conn = db.lock().unwrap();
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM turns WHERE persona_id = ?1 AND mode = 'NORMAL' AND excluded = 0",
        params![persona_id],
        |r| r.get(0),
    )?;
    Ok(n)
}

/// 当前生效模型版本：最近一次未回滚训练的版本号，默认 0（基座）
pub fn current_model_version(db: &Db) -> i64 {
    let conn = db.lock().unwrap();
    conn.query_row(
        "SELECT model_version FROM train_runs WHERE rolled_back = 0 ORDER BY id DESC LIMIT 1",
        [],
        |r| r.get(0),
    )
    .unwrap_or(0)
}
