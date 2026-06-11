// MV3 Service Worker：
// 1. 维护扩展图标角标（风格控制器在线状态，模块 B 降级提示的图标侧）。
// 2. 点击扩展图标 → 通知 content script 切换应急求助面板（模块 D 手动开关）。

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg && msg.type === "controller_state" && sender.tab) {
    chrome.action.setBadgeText({
      tabId: sender.tab.id,
      text: msg.online ? "" : "OFF"
    });
    if (!msg.online) {
      chrome.action.setBadgeBackgroundColor({ tabId: sender.tab.id, color: "#c92a2a" });
    }
  }
});

chrome.action.onClicked.addListener((tab) => {
  if (tab && tab.id != null) {
    chrome.tabs.sendMessage(tab.id, { type: "toggle_crisis_panel" }).catch(() => {});
  }
});
