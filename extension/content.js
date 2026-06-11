// ============================================================
// 编排层（模块 B + C + D 的运行时数据流，见第 3 节"单轮数据流"）
//
// 时序说明（为什么 turn N 要等到 turn N+1 才落库）：
//   - turn N 的 cloud_score 由"云端在第 N+1 条回复开头输出的 <score>"给出
//     （云端看到用户第 N+1 条消息后，评估自己第 N 条回复的安抚效果）；
//   - turn N 的隐式行为信号（regen/copied/followup_*）也只在用户对
//     第 N 条回复做出反应后才齐全。
//   - 因此 turn N 在「收到第 N+1 条回复」或「abandoned 超时」时 finalize 并 log_turn。
// ============================================================

(() => {
  "use strict";

  const adapter = GeminiAdapter; // TODO: 多站点时按 location.host 选择适配器
  let cfg = MC_CONFIG;

  // ---------- 离线角标（页面内 + 扩展图标） ----------
  function setControllerState(online) {
    try { chrome.runtime.sendMessage({ type: "controller_state", online }); } catch { /* SW 休眠等场景，忽略 */ }
    let badge = document.getElementById("mc-offline-badge");
    if (!online) {
      if (!badge) {
        badge = document.createElement("div");
        badge.id = "mc-offline-badge";
        badge.textContent = "风格控制器离线 — 已降级为原样发送";
        document.documentElement.appendChild(badge);
      }
    } else if (badge) {
      badge.remove();
    }
  }

  const ws = new MCWsClient({ ...cfg.ws, onStateChange: setControllerState });
  ws.connect();

  // ---------- 会话与轮次状态 ----------
  const sessionId = `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  let personaId = "default"; // TODO: 阶段 3 从后端 persona_config 获取
  let turnSeq = 0;

  // pendingTurn = 已提交、其 log_turn 尚未发送的上一轮
  let pendingTurn = null;
  let lastResponseDoneAt = null;
  let abandonTimer = null;

  function newTurn(userInput, stylePayload, degraded) {
    return {
      local_turn_seq: ++turnSeq,
      ts: new Date().toISOString(),
      session_id: sessionId,
      persona_id: personaId,
      user_input: userInput,
      style_payload: stylePayload, // 降级时为 null
      degraded: degraded ? 1 : 0,
      raw_cloud_output: null,
      clean_text: null,
      cloud_score: null,   // 由下一轮回复的 <score> 回填
      status: "normal",
      mode: "NORMAL", // NORMAL | CRITICAL
      regen_clicked: 0,
      copied: 0,
      followup_latency_ms: null,
      followup_len: null,
      abandoned: 0,
      format_miss: 0
    };
  }

  function finalizeTurn(turn, reason) {
    if (!turn || turn._finalized) return;
    turn._finalized = true;
    const { _finalized, local_turn_seq, ...fields } = turn;
    ws.send("log_turn", { ...fields, finalize_reason: reason });
  }

  function armAbandonTimer() {
    clearTimeout(abandonTimer);
    abandonTimer = setTimeout(() => {
      if (pendingTurn && !pendingTurn._finalized) {
        pendingTurn.abandoned = 1;
        finalizeTurn(pendingTurn, "abandoned");
        pendingTurn = null;
      }
    }, cfg.signals.abandonedAfterMs);
  }

  // ---------- 危机处理（模块 D / ADR-4） ----------
  // safeMode = 安全港模式：触发后立即"冻结日常人设"——不再向后端请求探索性风格，
  // 改为固定注入 cfg.safeStyle（零讽刺、100% 支持性安抚指令）。
  // 该模式下所有轮次标 CRITICAL（后端落库时强制 excluded=1，绝不进训练集）。
  // 仅当用户在面板中点击"恢复日常风格"才退出；关闭面板不退出。
  let safeMode = false;

  function localCrisisCheck(text) {
    const lower = (text || "").toLowerCase();
    return cfg.crisisKeywords.some(k => lower.includes(k.toLowerCase()));
  }

  function triggerCrisis(turn, source) {
    safeMode = true;
    if (turn) {
      turn.mode = "CRITICAL";
      turn.status = "crisis";
    }
    MCCrisisPanel.open(cfg.crisisResources, source, {
      onExitSafeMode: () => { safeMode = false; }
    });
    ws.send("crisis_event", { session_id: sessionId, source });
  }

  // ---------- 提交拦截 → style 注入（模块 B） ----------
  let currentTurn = null;

  adapter.interceptSubmit((rawText) => {
    // 用户的新一轮提交：先补齐上一轮的 followup 信号
    if (pendingTurn && !pendingTurn._finalized && lastResponseDoneAt) {
      pendingTurn.followup_latency_ms = Date.now() - lastResponseDoneAt;
      pendingTurn.followup_len = rawText.length;
    }
    clearTimeout(abandonTimer);

    // 本地危机二次校验（宁可误报）：命中即弹资源面板并进入安全港模式
    const crisisHit = localCrisisCheck(rawText);
    if (crisisHit && !safeMode) triggerCrisis(null, "local_rule");

    // ---- 安全港模式（模块 D）：冻结日常人设，固定注入安抚 style ----
    // 不请求后端（停止风格探索）；programmaticSubmit 纯 DOM 操作，离线也可用。
    if (safeMode) {
      currentTurn = newTurn(rawText, cfg.safeStyle, false);
      currentTurn.mode = "CRITICAL";
      currentTurn.status = "crisis";
      const styled = cfg.injectTemplate
        .replace("{style}", cfg.safeStyle)
        .replace("{userText}", rawText);
      adapter.programmaticSubmit(styled);
      return true; // 已接管
    }

    if (!ws.online) {
      // 降级：放行原文（返回 false = 不接管），但仍记录降级轮次
      currentTurn = newTurn(rawText, null, true);
      return false;
    }

    // 接管提交：异步请求 style，成功则注入改写文本，失败则原样程序化提交
    (async () => {
      let styled = rawText;
      let stylePayload = null;
      try {
        const resp = await ws.request("style_request", {
          session_id: sessionId,
          persona_id: personaId,
          user_input: rawText
        });
        stylePayload = resp.style_payload || cfg.placeholderStyle; // # MOCK 兜底
        styled = cfg.injectTemplate
          .replace("{style}", stylePayload)
          .replace("{userText}", rawText);
      } catch (e) {
        // 超时/后端错误 → 降级原样发送（模块 B 验收项）
        setControllerState(ws.online); // 刷新角标
      }
      currentTurn = newTurn(rawText, stylePayload, stylePayload === null);
      adapter.programmaticSubmit(styled);
    })();
    return true; // 已接管，阻断原生提交
  });

  // ---------- 流式输出采集 → 标签抽取/剥离（模块 C） ----------
  adapter.observeResponses({
    onChunk: (el) => {
      // 流式期间持续剥标签 + 遮蔽半截标签，避免用户看到度量块
      adapter.scrubElement(el, null, "mc-tag-streaming-mask");
    },
    onDone: (el, fullText) => {
      const parsed = MCTagParser.parse(fullText);
      adapter.scrubElement(el, null, "mc-tag-streaming-mask");

      // <score> 属于上一轮（见文件头时序说明）
      if (pendingTurn && !pendingTurn._finalized) {
        pendingTurn.cloud_score = parsed.score;
        finalizeTurn(pendingTurn, "next_response");
      }

      if (currentTurn) {
        currentTurn.raw_cloud_output = fullText;
        currentTurn.clean_text = parsed.cleanText;
        currentTurn.format_miss = parsed.formatMiss ? 1 : 0;
        if (parsed.status === "crisis") triggerCrisis(currentTurn, "tag");
        pendingTurn = currentTurn;
        currentTurn = null;
      }
      lastResponseDoneAt = Date.now();
      armAbandonTimer();
    }
  });

  // ---------- 隐式信号（第 6 节） ----------
  adapter.onRegenClicked(() => {
    if (pendingTurn && !pendingTurn._finalized) {
      pendingTurn.regen_clicked = 1;
      // 重新生成 = 强负信号，立即可定论，不必等下一轮
      finalizeTurn(pendingTurn, "regen");
      pendingTurn = null;
    }
  });

  adapter.onCopy(() => {
    if (pendingTurn && !pendingTurn._finalized) pendingTurn.copied = 1;
  });

  // 页面卸载前尽力 finalize（WS 可能已断，丢失可接受 —— 噪声容忍，第 7 节 #4）
  window.addEventListener("beforeunload", () => {
    if (pendingTurn && !pendingTurn._finalized) {
      pendingTurn.abandoned = 1;
      finalizeTurn(pendingTurn, "unload");
    }
  });

  // ---------- 手动应急开关（扩展图标点击 → SW 转发） ----------
  try {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg && msg.type === "toggle_crisis_panel") {
        if (MCCrisisPanel.isOpen()) MCCrisisPanel.close();
        else triggerCrisis(pendingTurn || currentTurn, "manual");
      }
    });
  } catch { /* 非扩展环境（单测）忽略 */ }

  // ---------- 配置覆盖（chrome.storage.local） ----------
  try {
    chrome.storage.local.get(["mc_config_override"], (res) => {
      if (res && res.mc_config_override) {
        cfg = deepMerge(cfg, res.mc_config_override);
      }
    });
  } catch { /* 非扩展环境忽略 */ }

  function deepMerge(base, over) {
    const out = { ...base };
    for (const k of Object.keys(over || {})) {
      out[k] = (over[k] && typeof over[k] === "object" && !Array.isArray(over[k]))
        ? deepMerge(base[k] || {}, over[k]) : over[k];
    }
    return out;
  }

  console.info("[MoodCompanion] content script loaded, session:", sessionId);
})();
