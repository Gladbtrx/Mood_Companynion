// ============================================================
// WebSocket 客户端（扩展 ↔ 本地后端，数据契约第 5 节）
// - 自动重连（指数退避）
// - request/response 关联：在消息上附加 req_id（契约的加法扩展，后端原样回传）
// - 超时 reject → 由编排层触发"离线降级放行"（模块 B）
// ============================================================

class MCWsClient {
  constructor({ url, token, requestTimeoutMs, reconnectBaseMs, reconnectMaxMs, onStateChange }) {
    this.url = url;
    this.token = token;
    this.requestTimeoutMs = requestTimeoutMs || 3000;
    this.reconnectBaseMs = reconnectBaseMs || 1000;
    this.reconnectMaxMs = reconnectMaxMs || 15000;
    this.onStateChange = onStateChange || (() => {});
    this._ws = null;
    this._online = false;
    this._retry = 0;
    this._pending = new Map(); // req_id -> {resolve, reject, timer}
    this._seq = 0;
    this._closedByUser = false;
  }

  get online() { return this._online; }

  connect() {
    this._closedByUser = false;
    this._open();
  }

  close() {
    this._closedByUser = true;
    if (this._ws) this._ws.close();
  }

  _open() {
    try {
      this._ws = new WebSocket(this.url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }
    this._ws.onopen = () => {
      this._retry = 0;
      // 首条消息做 token 校验（防其他本地进程误连，非安全机制）
      this._ws.send(JSON.stringify({ type: "auth", token: this.token }));
      this._setOnline(true);
    };
    this._ws.onmessage = (ev) => this._onMessage(ev);
    this._ws.onclose = () => {
      this._setOnline(false);
      this._failAllPending("ws_closed");
      if (!this._closedByUser) this._scheduleReconnect();
    };
    this._ws.onerror = () => { /* onclose 会跟着触发，统一在那里处理 */ };
  }

  _scheduleReconnect() {
    const delay = Math.min(this.reconnectBaseMs * Math.pow(2, this._retry++), this.reconnectMaxMs);
    setTimeout(() => { if (!this._closedByUser) this._open(); }, delay);
  }

  _setOnline(v) {
    if (this._online !== v) {
      this._online = v;
      this.onStateChange(v);
    }
  }

  _onMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.req_id && this._pending.has(msg.req_id)) {
      const p = this._pending.get(msg.req_id);
      this._pending.delete(msg.req_id);
      clearTimeout(p.timer);
      if (msg.type === "error") p.reject(new Error(msg.message || msg.code || "backend_error"));
      else p.resolve(msg);
    }
    // 无 req_id 的服务端推送暂不使用（保留扩展点）
  }

  _failAllPending(reason) {
    for (const [, p] of this._pending) {
      clearTimeout(p.timer);
      p.reject(new Error(reason));
    }
    this._pending.clear();
  }

  // 有应答请求：style_request 等
  request(type, payload) {
    return new Promise((resolve, reject) => {
      if (!this._online || !this._ws || this._ws.readyState !== WebSocket.OPEN) {
        reject(new Error("offline"));
        return;
      }
      const req_id = `r${++this._seq}_${Date.now()}`;
      const timer = setTimeout(() => {
        this._pending.delete(req_id);
        reject(new Error("timeout"));
      }, this.requestTimeoutMs);
      this._pending.set(req_id, { resolve, reject, timer });
      this._ws.send(JSON.stringify({ type, req_id, ...payload }));
    });
  }

  // 单向上报：log_turn / crisis_event（发不出去就静默丢弃并返回 false，不打扰用户）
  send(type, payload) {
    if (!this._online || !this._ws || this._ws.readyState !== WebSocket.OPEN) return false;
    try {
      this._ws.send(JSON.stringify({ type, ...payload }));
      return true;
    } catch {
      return false;
    }
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = MCWsClient;
}
