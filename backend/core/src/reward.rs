//! 奖励合成（第 6 节：纯隐式行为 + 云端自评分冷启动兜底，ADR-3）。
//! advantage 不在此计算 —— 它需要语义桶基线，由夜间训练管线（training/train.py）
//! 在训练前统一计算并回写 turns.advantage。

use crate::config::RewardConfig;
use crate::db::TurnLog;
use serde_json::json;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// 返回 (reward, components_json)。CRITICAL 轮次不应调用本函数（数据隔离）。
pub fn compute_reward(cfg: &RewardConfig, t: &TurnLog, n_behavior_samples: i64) -> (f64, String) {
    // engaged_followup：阈值时间内的实质性追问（第 6 节）
    let engaged = match (t.followup_len, t.followup_latency_ms) {
        (Some(len), Some(lat)) => len >= cfg.engaged_min_len && lat <= cfg.engaged_max_latency_ms,
        _ => false,
    };

    let z = cfg.w_copied * t.copied as f64
        + cfg.w_engaged * (engaged as i64) as f64
        - cfg.w_regen * t.regen_clicked as f64
        - cfg.w_abandoned * t.abandoned as f64;
    let r_behavior = sigmoid(z);

    // 冷启动混合：样本少时靠云端自评分，行为数据积累后行为信号接管
    let (reward, alpha, score_norm) = match t.cloud_score {
        Some(score) if (1..=5).contains(&score) => {
            let score_norm = (score - 1) as f64 / 4.0;
            let alpha = n_behavior_samples as f64 / (n_behavior_samples as f64 + cfg.k_cold_start);
            (alpha * r_behavior + (1.0 - alpha) * score_norm, alpha, Some(score_norm))
        }
        // 云端自评分缺失（format_miss 等）→ 完全用行为信号
        _ => (r_behavior, 1.0, None),
    };

    let components = json!({
        "r_behavior": r_behavior,
        "engaged_followup": engaged,
        "z": z,
        "alpha": alpha,
        "score_norm": score_norm,
        "n_behavior_samples": n_behavior_samples,
        "weights": { "w_copied": cfg.w_copied, "w_engaged": cfg.w_engaged,
                      "w_regen": cfg.w_regen, "w_abandoned": cfg.w_abandoned,
                      "k": cfg.k_cold_start }
    });
    (reward.clamp(0.0, 1.0), components.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_turn() -> TurnLog {
        TurnLog { session_id: "s".into(), persona_id: "p".into(), ..Default::default() }
    }

    #[test]
    fn regen_is_strong_negative() {
        let cfg = RewardConfig::default();
        let mut t = base_turn();
        t.regen_clicked = 1;
        let (r_neg, _) = compute_reward(&cfg, &t, 1000); // N 大 → 行为信号主导
        t.regen_clicked = 0;
        t.copied = 1;
        let (r_pos, _) = compute_reward(&cfg, &t, 1000);
        assert!(r_neg < 0.5 && r_pos > 0.5 && r_neg < r_pos);
    }

    #[test]
    fn cold_start_uses_cloud_score() {
        let cfg = RewardConfig::default();
        let mut t = base_turn();
        t.cloud_score = Some(5); // score_norm = 1.0
        let (r, _) = compute_reward(&cfg, &t, 0); // N=0 → alpha=0 → 全靠自评分
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn behavior_takes_over_with_samples() {
        let cfg = RewardConfig::default();
        let mut t = base_turn();
        t.cloud_score = Some(5);
        t.regen_clicked = 1; // 行为强负 vs 自评分满分
        let (r_cold, _) = compute_reward(&cfg, &t, 0);
        let (r_warm, _) = compute_reward(&cfg, &t, 10_000);
        assert!(r_warm < r_cold); // 样本积累后行为信号接管
    }

    #[test]
    fn missing_score_falls_back_to_behavior() {
        let cfg = RewardConfig::default();
        let t = base_turn();
        let (r, c) = compute_reward(&cfg, &t, 0);
        assert!((r - 0.5).abs() < 1e-9); // 全零信号 → sigmoid(0)
        assert!(c.contains("\"score_norm\":null"));
    }

    #[test]
    fn engaged_requires_both_thresholds() {
        let cfg = RewardConfig::default();
        let mut t = base_turn();
        t.followup_len = Some(100);
        t.followup_latency_ms = Some(cfg.engaged_max_latency_ms + 1); // 太迟
        let (r, _) = compute_reward(&cfg, &t, 1000);
        assert!((r - 0.5).abs() < 1e-2);
    }
}
