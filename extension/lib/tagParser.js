// ============================================================
// 隐藏标签容错解析器（模块 C / 数据契约第 5 节）
// 已知风险（第 7 节 #3）：云端可能不遵守格式或漂移。
// 设计铁律：本模块【绝不抛异常、绝不返回 undefined】。
// 标签缺失/非法 → score=null, status="normal", formatMiss=true，正文照常返回。
// ============================================================

const MCTagParser = (() => {
  const SCORE_RE = /<\s*score\s*>\s*([1-5])\s*<\/\s*score\s*>/i;
  const STATUS_RE = /<\s*status\s*>\s*(normal|crisis)\s*<\/\s*status\s*>/i;
  // 非法但成对的标签体（如 <score>9</score>、<status>panic</status>）整体剥除
  const ILLEGAL_PAIR_RE = /<\s*(score|status|style)\s*>[^<]{0,40}<\/\s*\1\s*>/gi;
  // 孤立/残缺标签（流式截断、云端漂移）
  const ORPHAN_TAG_RE = /<\/?\s*(score|status|style)\b[^>]*>/gi;
  // 流式过程中正文开头可能出现的"未闭合标签前缀"，用于渲染端临时遮蔽
  const STREAMING_PREFIX_RE = /^\s*<[a-z\s/]{0,8}$/i;

  function parse(raw) {
    if (typeof raw !== "string" || raw.length === 0) {
      return { score: null, status: "normal", cleanText: "", formatMiss: true };
    }
    let text = raw;
    let score = null;
    let status = "normal";
    let scoreHit = false;
    let statusHit = false;

    const sm = text.match(SCORE_RE);
    if (sm) {
      score = parseInt(sm[1], 10);
      scoreHit = true;
      text = text.replace(SCORE_RE, "");
    }
    const tm = text.match(STATUS_RE);
    if (tm) {
      status = tm[1].toLowerCase();
      statusHit = true;
      text = text.replace(STATUS_RE, "");
    }

    // 残片清理：先剥非法成对标签，再剥孤立标签
    text = text.replace(ILLEGAL_PAIR_RE, "").replace(ORPHAN_TAG_RE, "");

    const cleanText = text.replace(/^[\s:：、\-–—]+/, "").trimEnd();
    return {
      score,
      status,
      cleanText,
      formatMiss: !(scoreHit && statusHit)
    };
  }

  // 渲染端辅助：流式中开头是半个标签时返回 true（UI 暂时遮住首行，避免闪现）
  function looksLikeStreamingTagPrefix(s) {
    return STREAMING_PREFIX_RE.test(s || "");
  }

  return { parse, looksLikeStreamingTagPrefix };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = MCTagParser;
}
