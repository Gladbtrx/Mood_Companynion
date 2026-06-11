// ============================================================
// Gemini 站点适配器（ADR-6：站点逻辑全部收敛在此，后续可加 adapters/claude.js）
//
// ⚠️ 脆弱点声明（第 7 节 #1）：本文件 100% 依赖 Gemini 网页 DOM 结构，
// Google 改版即可能失效。缓解措施：
//   - 所有选择器集中在 SELECTORS，每项都是"按优先级排列的候选列表"；
//   - 时序全部用 MutationObserver / 轮询兜底，不硬编码 setTimeout 时点；
//   - 任何环节找不到节点都返回 null/false，由编排层降级放行（绝不阻断用户）。
// 选择器基于 2025 年末 Gemini 网页版结构（Quill contenteditable 输入框 +
// <model-response> 流式输出），失效时请按 docs/adapter-repair.md 排查更新。
// ============================================================

const GeminiAdapter = (() => {
  const SELECTORS = {
    input: [
      'div.ql-editor[contenteditable="true"]',
      'rich-textarea div[contenteditable="true"]',
      'div[contenteditable="true"][role="textbox"]'
    ],
    sendButton: [
      'button[aria-label="Send message"]',
      'button[aria-label*="发送"]',
      'button[aria-label*="Send"]',
      'button.send-button'
    ],
    // 流式进行中的"停止"按钮：存在 = 仍在生成
    stopButton: [
      'button[aria-label*="Stop"]',
      'button[aria-label*="停止"]'
    ],
    responseContainer: [
      'model-response',
      'div.model-response-text',
      'message-content'
    ],
    regenButton: [
      'button[aria-label*="Regenerate"]',
      'button[aria-label*="重新生成"]',
      'button[aria-label*="Redo"]'
    ],
    copyButton: [
      'button[aria-label*="Copy"]',
      'button[aria-label*="复制"]'
    ]
  };

  const STABLE_MS = 800;   // 文本静止 + 停止按钮消失 达 800ms → 视为流式结束
  const POLL_MS = 250;

  function q(cands, root) {
    for (const sel of cands) {
      const el = (root || document).querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  function qa(cands, root) {
    for (const sel of cands) {
      const els = (root || document).querySelectorAll(sel);
      if (els.length) return Array.from(els);
    }
    return [];
  }

  // ---------- 输入框 ----------
  function getInputBox() { return q(SELECTORS.input); }

  function readInput() {
    const box = getInputBox();
    return box ? box.innerText.trim() : "";
  }

  // ⚠️ 脆弱点：Quill 编辑器不接受直接改 textContent 后的状态同步，
  // 这里用"清空 + execCommand insertText"，并补发 input 事件。
  function writeInput(text) {
    const box = getInputBox();
    if (!box) return false;
    box.focus();
    document.execCommand("selectAll", false, null);
    document.execCommand("delete", false, null);
    const ok = document.execCommand("insertText", false, text);
    if (!ok || box.innerText.trim() === "") {
      // 兜底路径：直接写 DOM 再补事件（部分 Quill 版本可接受）
      box.innerHTML = "";
      const p = document.createElement("p");
      p.textContent = text;
      box.appendChild(p);
      box.dispatchEvent(new InputEvent("input", { bubbles: true, data: text }));
    }
    return true;
  }

  function clickSend() {
    const btn = q(SELECTORS.sendButton);
    if (btn && !btn.disabled) {
      btn.click();
      return true;
    }
    // 兜底：对输入框模拟 Enter
    const box = getInputBox();
    if (box) {
      box.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Enter", code: "Enter", keyCode: 13, bubbles: true
      }));
      return true;
    }
    return false;
  }

  // ---------- 提交拦截（模块 B 入口） ----------
  // handler(rawText) 返回 true 表示"已接管"（适配器阻断原生提交），
  // 返回 false 表示放行（降级路径）。
  // 用 capture 阶段保证先于站点自身的监听器执行。
  let _suppressIntercept = false; // 程序化提交时置位，避免拦截自己

  function interceptSubmit(handler) {
    document.addEventListener("keydown", (ev) => {
      if (_suppressIntercept) return;
      if (ev.key !== "Enter" || ev.shiftKey || ev.isComposing) return;
      const box = getInputBox();
      if (!box || !box.contains(ev.target)) return;
      const raw = readInput();
      if (!raw) return;
      if (handler(raw)) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
      }
    }, true);

    document.addEventListener("click", (ev) => {
      if (_suppressIntercept) return;
      const btn = ev.target && ev.target.closest && ev.target.closest("button");
      if (!btn) return;
      if (!matchesAny(btn, SELECTORS.sendButton)) return;
      const raw = readInput();
      if (!raw) return;
      if (handler(raw)) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
      }
    }, true);
  }

  function matchesAny(el, cands) {
    return cands.some(sel => { try { return el.matches(sel); } catch { return false; } });
  }

  // 程序化提交（注入改写后的文本并发送）
  function programmaticSubmit(text) {
    _suppressIntercept = true;
    try {
      if (!writeInput(text)) return false;
      // 等 Quill 状态同步后点发送；用轮询而非固定延时
      let tries = 0;
      const timer = setInterval(() => {
        tries++;
        if (clickSend() || tries > 12) {
          clearInterval(timer);
          // 留一拍再放开拦截，避免点击事件冒泡回拦截器
          setTimeout(() => { _suppressIntercept = false; }, 300);
        }
      }, POLL_MS);
      return true;
    } catch (e) {
      _suppressIntercept = false;
      return false;
    }
  }

  // ---------- 流式输出采集（模块 C 入口） ----------
  // 监听新增的回复容器；流式期间回调 onChunk(el, text)，结束回调 onDone(el, fullText)。
  function observeResponses({ onChunk, onDone }) {
    const known = new WeakSet();
    const tracked = []; // {el, lastText, lastChange, doneFired}

    const scan = () => {
      for (const el of qa(SELECTORS.responseContainer)) {
        if (!known.has(el)) {
          known.add(el);
          tracked.push({ el, lastText: "", lastChange: Date.now(), doneFired: false });
        }
      }
    };

    const mo = new MutationObserver(() => scan());
    mo.observe(document.body, { childList: true, subtree: true });
    scan();

    setInterval(() => {
      scan();
      const now = Date.now();
      const streaming = !!q(SELECTORS.stopButton);
      for (const t of tracked) {
        if (t.doneFired || !t.el.isConnected) continue;
        const text = t.el.innerText || "";
        if (text !== t.lastText) {
          t.lastText = text;
          t.lastChange = now;
          if (onChunk) onChunk(t.el, text);
        } else if (text && !streaming && now - t.lastChange >= STABLE_MS) {
          t.doneFired = true;
          if (onDone) onDone(t.el, text);
        }
      }
    }, POLL_MS);
  }

  // 把已渲染回复里的隐藏标签从 DOM 中剥掉（只动文本节点，保留其余结构）
  // cleanFn: rawText -> cleanText（由 tagParser 提供）
  function scrubElement(el, cleanFn, maskClass) {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let node;
    const tagRe = /<\/?\s*(score|status|style)\b[^>]*>|<\s*(score|status)\s*>[^<]{0,40}<\/\s*\2\s*>/gi;
    while ((node = walker.nextNode())) {
      if (tagRe.test(node.textContent)) {
        node.textContent = node.textContent.replace(tagRe, "");
      }
      tagRe.lastIndex = 0;
      // 流式中开头是半截标签 → 给最近的块级父元素加遮蔽类
      if (MCTagParser.looksLikeStreamingTagPrefix(node.textContent)) {
        const blk = node.parentElement;
        if (blk && maskClass) blk.classList.add(maskClass);
      } else if (node.parentElement && maskClass) {
        node.parentElement.classList.remove(maskClass);
      }
    }
  }

  // ---------- 隐式信号钩子（第 6 节） ----------
  function onRegenClicked(cb) {
    document.addEventListener("click", (ev) => {
      const btn = ev.target && ev.target.closest && ev.target.closest("button");
      if (btn && matchesAny(btn, SELECTORS.regenButton)) cb();
    }, true);
  }

  function onCopy(cb) {
    // 两条路径：点回复工具栏的复制按钮，或对回复区文本执行复制
    document.addEventListener("click", (ev) => {
      const btn = ev.target && ev.target.closest && ev.target.closest("button");
      if (btn && matchesAny(btn, SELECTORS.copyButton)) cb();
    }, true);
    document.addEventListener("copy", () => {
      const sel = document.getSelection();
      if (!sel || sel.isCollapsed) return;
      const anchor = sel.anchorNode && sel.anchorNode.parentElement;
      if (anchor && anchor.closest && SELECTORS.responseContainer.some(s => anchor.closest(s))) cb();
    }, true);
  }

  return {
    name: "gemini",
    SELECTORS,
    getInputBox, readInput, writeInput, clickSend,
    interceptSubmit, programmaticSubmit,
    observeResponses, scrubElement,
    onRegenClicked, onCopy
  };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = GeminiAdapter;
}
