# VocabTool 网站版

在保留原有 Streamlit 应用不变的前提下，新增的 Web + PWA 版本。

## 目录结构

```
vocabtool/
├── app.py, vocab.py, ...    # 原有 Streamlit 应用（未修改）
├── backend/                  # FastAPI 后端（新增）
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── ...
├── frontend/                 # React 前端（新增）
│   ├── src/
│   ├── public/
│   └── ...
└── coca_cleaned.csv          # 词库（与 Streamlit 共用）
```

## 本地运行

### 1. 后端

```bash
cd backend
pip install -r requirements.txt
# 复制 .env.example 为 .env，配置 OPENAI_API_KEY 等
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 文档：http://localhost:8000/docs

### 2. 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器打开：http://localhost:5173

前端通过 Vite 代理将 `/api` 转发到后端，无需单独配置 `VITE_API_URL`。

### 3. 环境变量（backend/.env）

- `OPENAI_API_KEY` — OpenAI API 密钥（查词、Anki 生成必填）
- `OPENAI_BASE_URL` — 可选，默认 https://api.openai.com/v1
- `OPENAI_MODEL` — 可选，默认 gpt-4o-mini

## PWA

- `manifest.json` 与 `service-worker.js` 已配置
- 如需完整支持「添加到主屏幕」，请在 `frontend/public/` 下添加：
  - `icon-192.png`（192×192）
  - `icon-512.png`（512×512）

## 生产部署

- 前端：`npm run build` 后部署 dist/ 到静态托管
- 后端：`gunicorn -w 4 -b 0.0.0.0:8000 main:app` 或 Docker
