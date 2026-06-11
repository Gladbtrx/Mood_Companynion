# Mood Companion — 本地风格控制器（课程演示项目）

把闭源云端 LLM（首期 **Gemini** 网页版）的回复风格，从"专业/学术/说教"引导为
高适配度的情感陪伴：**浏览器扩展**拦截输入并注入本地小模型生成的 `<style>` 风格指令，
**本地后端**记录隐式行为奖励，**夜间 RWR 训练**让人设随使用进化。

> ⚠️ **定位声明**：本项目是课程作业 / 个人本机演示，目标是跑通闭环并展示架构，
> 不是生产级安全系统，不做规模化或对外分发。

## 隐私口径（精确表述，不得夸大）

本系统**不能**让对话对云端 LLM 服务商不可见——用户输入（经改写后）仍会发送到云端。
本地化的是**个性化策略、奖励信号与训练过程**：人格配置、行为日志、奖励数据、
模型微调全部不出本机。

## 架构总览

```
┌─ 浏览器扩展 (MV3) ───────────────────────────────┐
│ content.js 编排 │ adapters/gemini.js 站点适配     │
│ tagParser 容错抽取 <score>/<status> │ crisisPanel │
└────────────┬────────────────────────────────────┘
             │ ws://127.0.0.1:8765（简单 token 校验）
┌────────────┴────────────────────────────────────┐
│ 本地后端 (Rust, Tauri 壳可选 / headless 可独立跑)  │
│ WS 服务 │ 人格引擎 │ 风格生成(Qwen via Ollama)     │
│ SQLite 落库 + 奖励合成                            │
├──────────────────────────────────────────────────┤
│ 夜间 RWR 训练引擎 (Python, 02:00–06:00 空闲触发)   │
│ 读 SQLite → 加权样本 → QLoRA → GGUF → ollama 更新 │
└──────────────────────────────────────────────────┘
```

单轮数据流、模块规格与 ADR 见 `docs/phase4-training.md` 与各源文件头注释。

## 本地使用方法

### 0. 前置条件

| 组件 | 用途 | 必需性 |
|---|---|---|
| Chrome / Edge | 加载 MV3 扩展 | 必需 |
| Rust 工具链（`cargo`） | 编译本地后端 | 必需 |
| [Ollama](https://ollama.com) + `qwen2.5:1.5b-instruct` | 本地风格生成 | 必需（缺时扩展自动降级放行原文） |
| Python 3.10+ | 夜间训练管线 | 训练阶段需要（纯标准库即可跑 mock） |
| NVIDIA GPU + `pip install -r training/requirements.txt` | 真 QLoRA 训练 | 可选（无则自动 mock） |

### 1. 启动本地模型与后端

```bash
ollama serve                          # 终端 A
ollama pull qwen2.5:1.5b-instruct

cd mood-companion                     # 终端 B
cargo run --bin mood-backend          # headless 后端，监听 ws://127.0.0.1:8765
# （可选 GUI 壳：cd backend && cargo tauri dev —— 需系统 WebView 依赖）
```

后端配置集中在 `config/backend.json`（端口、token、模型名、奖励权重、训练超参）。

### 2. 安装浏览器扩展

1. 打开 `chrome://extensions`，开启"开发者模式"；
2. "加载已解压的扩展程序" → 选择 `mood-companion/extension/` 目录；
3. 打开 https://gemini.google.com —— 右下角无红色"离线"角标即说明已连上后端。

### 3. 创建人格（模块 A，冷启动）

后端运行时，用任一 WS 客户端发一条 `persona_create`（或直接复用冒烟客户端源码）：

```json
{"type":"auth","token":"mood-companion-dev-token"}
{"type":"persona_create","req_id":"p1","one_liner":"一个嘴上嫌弃我、心里其实很在乎我的青梅竹马"}
```

生成的 `persona_config.json` 落在 `data/personas/`；校验失败自动重试 ≤2 次后回退默认模板。

### 4. 日常使用

直接在 Gemini 网页正常聊天即可，扩展全程无感：

- 你的输入会被拼上本地生成的 `<style>` 指令与隐藏度量指令后自动提交；
- 云端回复开头的 `<score>/<status>` 标签被瞬间抽取剥离，你只看到干净正文；
- 复制/重新生成/追问等隐式行为连同自评分一起落入本地 `data/mood.db`。

**离线降级**：后端没起或超时，扩展显示"风格控制器离线"角标并放行原文，绝不阻断使用。

**安全港模式（模块 D）**：云端标 `crisis`、本地关键词命中、或点击扩展图标手动触发时，
弹出真实求助资源面板（明确标注"演示功能，不能替代专业帮助"），同时冻结日常人设、
改注入 100% 支持性安抚语气；这些轮次标 `CRITICAL`，**绝不进训练集**。
面板里点"恢复日常风格"才退出该模式。

### 5. 冒烟测试与验收

```bash
cargo run --bin smoke                          # 端到端：auth→style→log→crisis→persona
cargo test -p mood-backend-core                # Rust 单测（奖励合成、style 清洗）
node extension/test/tagParser.test.js          # 标签容错解析 11 例
python3 -m training.dataset --selfcheck        # 训练/运行时 prompt 逐字一致性
python3 -m training.train --mock --force       # RWR 主线 mock 全回环
```

### 6. 夜间训练（模块 E）

```bash
python3 -m training.train --mock --force   # 无 GPU 演示完整回环
bash scripts/install-cron.sh               # 注册每日 02:05 唤醒（窗口/空闲/样本三重门控）
```

主线是 **RWR**（ADR-2）；`train_dpo.py` 是标注了"近似、有偏"的选做对比支路。
详见 `training/README.md`。

## 已知风险与边界（如实声明）

1. **DOM 注入脆弱**：依赖 Gemini 网页结构，改版即可能失效；选择器集中在
   `extension/adapters/gemini.js`，失效时降级放行。
2. **合规边界**：自动化操作与隐藏指令注入处于第三方 ToS 灰色地带；仅限个人本机
   课程演示，不得规模化或分发。
3. **隐藏标签不可靠**：解析全容错，缺失时 score=null/status=normal，正文照常渲染。
4. **隐式奖励有噪声**：引入云端自评分冷启动、advantage 去偏与训练回滚护栏。
5. **危机检测不可靠**：仅演示，导向真实求助资源，不作安全网（ADR-4）。
6. **隐私**：见上方隐私口径。

## 目录结构

```
extension/   MV3 扩展（编排、站点适配器、标签解析、危机面板、WS 客户端）
backend/     Rust 后端（core = headless 可独立运行；src-tauri = 可选 GUI 壳）
training/    Python 夜间训练（train.py 主线 RWR；train_dpo.py 选做支路；调度器）
config/      backend.json（Rust 与 Python 共用单一配置）
data/        SQLite 库、人格配置、适配器/数据集产物（全部本地）
scripts/     cron 安装脚本
docs/        设计取舍文档
```
