// ============================================================
// 紧急安全港面板（模块 D / ADR-4）
// 触发源：云端 <status>crisis</status> | 本地关键词命中 | 用户手动开关。
// 铁律：
//   1. 面板必须清晰标注"非 AI、演示功能、不能替代专业帮助"。
//   2. 展示区域可配置的真实求助资源。
//   3. 本模块只做"导向真实资源"，不做任何自动处置。
// ============================================================

const MCCrisisPanel = (() => {
  const PANEL_ID = "mc-crisis-panel";

  function isOpen() {
    return !!document.getElementById(PANEL_ID);
  }

  function close() {
    const el = document.getElementById(PANEL_ID);
    if (el) el.remove();
  }

  /**
   * @param {object} resourcesCfg MC_CONFIG.crisisResources
   * @param {string} source "tag" | "local_rule" | "manual"
   * @param {object} [opts] { onExitSafeMode } 用户点击"恢复日常风格"时回调（模块 D 安全港出口）
   */
  function open(resourcesCfg, source, opts) {
    if (isOpen()) return;
    const region = resourcesCfg.region || "INTL";
    const list = resourcesCfg[region] || resourcesCfg.INTL || [];
    const intl = region === "INTL" ? [] : (resourcesCfg.INTL || []);

    const overlay = document.createElement("div");
    overlay.id = PANEL_ID;
    overlay.innerHTML = `
      <div class="mc-crisis-card" role="alertdialog" aria-label="求助资源">
        <div class="mc-crisis-header">⚠️ 求助资源（这不是 AI 生成的内容）</div>
        <div class="mc-crisis-body">
          <p class="mc-crisis-disclaimer">
            本面板由一个<strong>不可靠的演示功能</strong>触发（来源：${escapeHtml(source)}），
            可能误判，也可能漏判。<strong>它不能替代专业帮助或紧急服务。</strong>
            如果你或他人正处于危险中，请立即拨打当地紧急电话或下方热线。
          </p>
          <ul class="mc-crisis-list">
            ${list.map(r => `<li><strong>${escapeHtml(r.name)}</strong>：<span class="mc-crisis-contact">${escapeHtml(r.contact)}</span> <em>${escapeHtml(r.note || "")}</em></li>`).join("")}
            ${intl.map(r => `<li><strong>${escapeHtml(r.name)}</strong>：<span class="mc-crisis-contact">${escapeHtml(r.contact)}</span> <em>${escapeHtml(r.note || "")}</em></li>`).join("")}
          </ul>
          <p class="mc-crisis-note">号码可能变动，请以官方公布为准。已切换到固定的温和安抚语气（日常人设暂停），
            这些对话被排除出本地训练数据。关闭面板后安抚语气仍然保持，直到你点击"恢复日常风格"。</p>
        </div>
        <button class="mc-crisis-close" type="button">我已了解，关闭面板（保持安抚语气）</button>
        <button class="mc-crisis-exit-safe" type="button">恢复日常风格</button>
      </div>`;
    overlay.querySelector(".mc-crisis-close").addEventListener("click", close);
    overlay.querySelector(".mc-crisis-exit-safe").addEventListener("click", () => {
      close();
      if (opts && typeof opts.onExitSafeMode === "function") opts.onExitSafeMode();
    });
    document.documentElement.appendChild(overlay);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  return { open, close, isOpen };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = MCCrisisPanel;
}
