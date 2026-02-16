# Vocab Flow Ultra

一个面向英语学习者的 Streamlit 工具：从文本中提取重点词，快速查词，并一键生成带语音的 Anki 卡组。

## 主要功能

### 1) 快速查词（内置 DeepSeek/OpenAI 兼容接口）
- 输入单词或短语，返回精炼释义、词源、例句
- 支持续查（点击结果中的英文词）
- 带词频 rank 标识

### 2) 生成重点单词
- `2.1 粘贴文本`：从粘贴内容提取重点词（按 rank 过滤）
- `2.2 文章链接`：抓取网页正文后提词（按 rank 过滤）
- `2.3 上传文件`：支持 TXT / PDF / DOCX / EPUB / CSV / Excel / DB / SQLite
- `2.4 词库生成`：按 rank 顺序生成或随机抽取
- `2.5 粘贴整理词表（不筛 rank）`

所有词表生成后，统一支持：
- 一键复制词表
- 内置 AI 一键制卡（可选语音）
- 生成可复制给第三方 AI 的 Prompt（可选卡片格式）

### 3) 粘贴 AI 内容制卡（语音）
- 粘贴第三方 AI 输出，解析为卡片并导出 `.apkg`
- 支持发音人选择与批量语音合成

## 技术栈

- `streamlit`（UI）
- `openai`（兼容 OpenAI 协议）
- `nltk + lemminflect`（词形还原与词汇分析）
- `edge-tts`（语音）
- `genanki`（Anki 打包）
- `pypdf / python-docx / ebooklib / bs4`（多格式文本抽取）

## 项目结构

- `app.py`：主入口与页面编排
- `ai.py`：AI 查词、批量制卡 Prompt 构造与调用
- `vocab.py`：词汇分析、rank 过滤
- `extraction.py`：多格式内容提取
- `anki_parse.py`：AI 输出解析
- `anki_package.py`：Anki 打包与语音文件处理
- `resources.py`：词库/NLP 资源加载
- `constants.py`：全局常量
- `config.py`：配置读取（`st.secrets`）

## 环境要求

- Python 3.10+（推荐）
- 依赖见 `requirements.txt`

安装依赖：

```bash
pip install -r requirements.txt
```

## 配置说明

在 `.streamlit/secrets.toml` 中配置：

```toml
OPENAI_API_KEY = "your-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可替换为兼容网关
OPENAI_MODEL = "deepseek-chat"
```

## 运行方式

```bash
streamlit run app.py
```

默认地址：`http://localhost:8501`

## 词库数据文件

项目启动时会读取 CSV 词库文件：
- `coca_cleaned.csv`（推荐）
- `data.csv` / `vocab.csv`（兼容兜底）

若上述文件都缺失，页面会提示词库数据缺失。

## 测试

```bash
pytest -q
```

## 说明与建议

- 内置 AI 制卡有上限（当前为 `MAX_AUTO_LIMIT`），大批量建议走第三方 Prompt 分批
- 首次运行会准备 NLP 资源，启动可能略慢
- 语音生成数量较大时，耗时主要取决于网络与 TTS 并发

## License

仅供学习与个人项目使用。若用于商用，请自行补充许可证与第三方服务条款说明。

