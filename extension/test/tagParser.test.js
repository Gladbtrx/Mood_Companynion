// 阶段 1 单元测试：隐藏标签容错解析（node --test 运行，无第三方依赖）
const { test } = require("node:test");
const assert = require("node:assert");
const TagParser = require("../lib/tagParser.js");

test("标准格式：抽取并剥离 score/status", () => {
  const r = TagParser.parse("<score>4</score><status>normal</status> 哼，谁担心你了，快去睡觉。");
  assert.strictEqual(r.score, 4);
  assert.strictEqual(r.status, "normal");
  assert.strictEqual(r.cleanText, "哼，谁担心你了，快去睡觉。");
  assert.strictEqual(r.formatMiss, false);
});

test("crisis 状态被识别", () => {
  const r = TagParser.parse("<score>2</score><status>crisis</status>别这样说，我在呢。");
  assert.strictEqual(r.status, "crisis");
  assert.strictEqual(r.cleanText, "别这样说，我在呢。");
});

test("标签缺失：缺省值 + formatMiss，正文不受影响", () => {
  const r = TagParser.parse("今天也辛苦啦，早点休息。");
  assert.strictEqual(r.score, null);
  assert.strictEqual(r.status, "normal");
  assert.strictEqual(r.cleanText, "今天也辛苦啦，早点休息。");
  assert.strictEqual(r.formatMiss, true);
});

test("只有 score 没有 status：formatMiss=true 但 score 仍生效", () => {
  const r = TagParser.parse("<score>5</score>真棒！");
  assert.strictEqual(r.score, 5);
  assert.strictEqual(r.formatMiss, true);
  assert.strictEqual(r.cleanText, "真棒！");
});

test("格式漂移：大小写/空白容错", () => {
  const r = TagParser.parse("< Score > 3 </ score >< STATUS >Normal</ status >好的呀");
  assert.strictEqual(r.score, 3);
  assert.strictEqual(r.status, "normal");
  assert.strictEqual(r.cleanText, "好的呀");
  assert.strictEqual(r.formatMiss, false);
});

test("非法值：score 越界与未知 status 被剥除且不污染正文", () => {
  const r = TagParser.parse("<score>9</score><status>panic</status>没事的。");
  assert.strictEqual(r.score, null);
  assert.strictEqual(r.status, "normal");
  assert.strictEqual(r.formatMiss, true);
  assert.strictEqual(r.cleanText, "没事的。");
});

test("孤立残缺标签被清理", () => {
  const r = TagParser.parse("</score><status>正文从这里开始。");
  assert.ok(!r.cleanText.includes("<"));
  assert.ok(r.cleanText.includes("正文从这里开始"));
});

test("云端把 style 标签回显时也剥掉", () => {
  const r = TagParser.parse("<score>4</score><status>normal</status><style>毒舌</style>嗯哼。");
  assert.strictEqual(r.cleanText, "嗯哼。");
});

test("空输入/非字符串不抛异常", () => {
  assert.doesNotThrow(() => TagParser.parse(""));
  assert.doesNotThrow(() => TagParser.parse(null));
  assert.doesNotThrow(() => TagParser.parse(undefined));
  assert.strictEqual(TagParser.parse(null).formatMiss, true);
});

test("流式半截标签前缀识别", () => {
  assert.ok(TagParser.looksLikeStreamingTagPrefix("<sco"));
  assert.ok(TagParser.looksLikeStreamingTagPrefix("<"));
  assert.ok(!TagParser.looksLikeStreamingTagPrefix("正常文本"));
  assert.ok(!TagParser.looksLikeStreamingTagPrefix("<p>html"));
});

test("多行真实样例：度量块独立成行", () => {
  const raw = "<score>5</score><status>normal</status>\n哼，就知道你又熬夜了。\n\n快去睡，明天我还要听你汇报呢。";
  const r = TagParser.parse(raw);
  assert.strictEqual(r.score, 5);
  assert.ok(r.cleanText.startsWith("哼"));
  assert.ok(r.cleanText.includes("汇报"));
});
