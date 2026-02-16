# VocabTool 项目现状与下一步行动

## 一、项目现状

### 1. 当前代码结构

| 位置 | 形态 | 说明 |
|------|------|------|
| **D:\project\vocabtool** | Streamlit 应用 | 完整的业务逻辑，但**不是** PWA/网站 |
| **D:\download\vocabtool_project_plan** | React + FastAPI 脚手架 | 有 PWA 结构，但**大量 Stub** |

### 2. 目标架构（项目计划）

```
backend/  — FastAPI 后端（查词、提取、Anki、TTS）
frontend/ — React 19 + TypeScript + Tailwind 前端
PWA 支持 — manifest.json + service-worker.js
```

---

## 二、代码问题与缺失

### 2.1 vocabtool（Streamlit）— 逻辑完整

✅ **已有完整实现：**
- `vocab.py` — 基于 COCA 词频 + NLTK/lemminflect 的 `analyze_logic`
- `ai.py` — OpenAI 查词 `get_word_quick_definition`、批量生成 `process_ai_in_batches`
- `extraction.py` — 支持 PDF、DOCX、EPUB、URL、CSV、Excel、SQLite、Anki 导出
- `anki_parse.py` — 解析 AI 返回的 `|||` 格式
- `anki_package.py` — genanki 真实生成 .apkg + TTS
- `tts.py` — edge-tts 异步批量合成
- `resources.py` — 加载 coca_cleaned.csv / vocab.pkl

❌ **问题：**
- 依赖 `streamlit`、`st.session_state`、`st.error` 等，无法直接作为 Web API 使用
- 配置来自 `.streamlit/secrets.toml`，非 FastAPI 风格
- 不是 PWA，无法安装到主屏幕

### 2.2 vocabtool_project_plan（目标 Web 架构）— 大量 Stub

| 模块 | 状态 | 问题 |
|------|------|------|
| **vocab_service** | ❌ 简化版 | 用 `Counter` 模拟词频，无 COCA、无 NLTK |
| **ai_service** | ❌ Stub | 返回假数据，未接入 OpenAI |
| **anki_service** | ❌ Stub | 生成占位 .apkg（PK\x03\x04），未用 genanki |
| **tts_service** | ❌ Stub | 空 mp3，未接入 edge-tts |
| **file_handler** | ⚠️ 部分 | 支持 PDF/DOCX/TXT，缺 EPUB、Excel、CSV、SQLite、Anki 导出 |
| **extraction 路由** | ⚠️ 简化 | 无 Anki txt 解析、无词形还原 |
| **QueryWord API** | ⚠️ 模型不匹配 | 返回 `definition`+`example`，原版有词源、rank |
| **GenerateAnki API** | ⚠️ 流程不完整 | 接收 words 后应：AI 生成 → 解析 → genanki+TTS，当前未实现 |

### 2.3 PWA 相关

- `manifest.json` 已配置，但依赖 `icon-192.png` 和 `icon-512.png`，需自行准备
- `service-worker.js` 为简单缓存策略，可工作

---

## 三、你现在应该做的事

### 方案 A：整合并迁移（推荐）

**目标：** 把 vocabtool 的完整功能迁移到 Web 架构，形成真正的 PWA 网站。

#### 步骤 1：整合项目结构

在 `D:\project\vocabtool` 下建立：

```
D:\project\vocabtool\
├── backend/           # 新建，从 project_plan 复制并增强
├── frontend/          # 新建，从 project_plan 复制
├── app.py             # 保留（Streamlit 版可继续用）
├── vocab.py           # 保留，供 backend 复用
├── extraction.py      # 保留
├── ai.py              # 需改写（去掉 Streamlit 依赖）
├── anki_parse.py      # 保留
├── anki_package.py    # 保留
├── tts.py             # 保留
├── resources.py       # 需改写（去掉 Streamlit 依赖）
├── config.py          # 需改写（支持 .env）
├── constants.py       # 保留
├── coca_cleaned.csv   # 词库
└── ...
```

#### 步骤 2：后端迁移任务

| 任务 | 说明 |
|------|------|
| 1. 将 `backend/` 从 project_plan 复制到 vocabtool | 保持 FastAPI 结构 |
| 2. `vocab_service` | 调用 `vocab.analyze_logic`，使用 COCA 词库 |
| 3. `ai_service` | 调用 `ai.get_word_quick_definition`、`process_ai_in_batches`（需去 Streamlit） |
| 4. `anki_service` | 调用 `ai.process_ai_in_batches` → `anki_parse.parse_anki_data` → `anki_package.generate_anki_package` |
| 5. `tts_service` | 调用 `tts.run_async_batch` 或直接使用 edge-tts |
| 6. `file_handler` | 复用 `extraction.py` 的 PDF/DOCX/EPUB/Excel/CSV 等逻辑 |
| 7. 增加 Anki txt 导出解析 | 复用 `extraction.parse_anki_txt_export` |
| 8. 配置 | 用 pydantic-settings 读取 `.env`（OPENAI_API_KEY 等） |

#### 步骤 3：前端调整（可选）

- 查词页：支持显示词源、rank（若 API 返回）
- 提取页：支持「从 Anki 导出导入」
- 提取 → Anki：提取结果可直接跳转到 Anki 页面并带参

#### 步骤 4：PWA 完善

- 在 `frontend/public/` 添加 `icon-192.png`、`icon-512.png`
- 测试「添加到主屏幕」

---

### 方案 B：保留 Streamlit，仅加 PWA 外壳（不推荐）

Streamlit 本身不支持 PWA，需反向代理等复杂方案，且体验不如原生 React。

---

## 四、优先级建议

1. **高优先级：** 将 backend 从 project_plan 复制到 vocabtool，并实现 `vocab_service`、`ai_service`、`anki_service`、`tts_service` 的真实逻辑。
2. **中优先级：** 完善 `file_handler`（EPUB、Excel 等），增加 Anki 导出解析。
3. **低优先级：** 查词 API 返回词源/rank，前端展示优化。

---

## 五、快速验证命令

```bash
# 后端（在 vocabtool 根目录或 backend 目录）
cd D:\project\vocabtool
pip install -r requirements.txt
# 若使用 backend：cd backend && uvicorn main:app --reload --port 8000

# 前端（在 project_plan 或整合后的 frontend）
cd D:\download\vocabtool_project_plan\frontend
npm install
npm run dev
# 浏览器打开 http://localhost:5173
```

---

## 六、总结

- **当前：** vocabtool 是功能完整的 Streamlit 应用，project_plan 是 Web 架构骨架但多为占位实现。
- **目标：** 做一个 PWA 网站，功能等同于 vocabtool。
- **建议：** 在 vocabtool 项目中引入 backend + frontend 结构，把现有 Python 逻辑迁移到 FastAPI services，并补齐 PWA 图标。完成迁移后即可拥有完整的 Web + PWA 版本。
