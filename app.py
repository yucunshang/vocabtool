# app.py
import streamlit as st
import re
import time
from collections import Counter
import io
import csv
from datetime import datetime
import os

# =====================================
# 页面配置 & 基本样式（让界面更像手机 App）
# =====================================
st.set_page_config(page_title="Vocab Master — Mobile", layout="wide")

# 小的 CSS 改造，让 UI 更像手机 app（按钮更大、字体更大）
st.markdown("""
<style>
/* container */
.block-container {
  padding: 12px;
  max-width: 640px;
  margin: auto;
}

/* large buttons */
.stButton>button {
  height: 48px;
  font-size: 16px;
}

/* larger text area */
textarea {
  font-size: 15px !important;
}

/* code box */
.stCodeBlock pre {
  font-size: 14px !important;
}

/* responsive: single column on narrow screens */
@media (max-width: 740px) {
  .css-1d391kg { padding: 6px; } /* st container internal class can vary */
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Vocab Master — 手机友好版")
st.caption("保留文本筛词 → 生成 Prompt → 在手机端用 AI 制卡并导入 Anki 的完整流程（不含内置 API）")

# =====================================
# 词库加载（替换为你的 COCA / 本地词库文件）
# 如果你已有 data/coca_cleaned.csv 或 data.csv，可加载真实数据
# 这里用示例小词库作为 fallback
# =====================================
@st.cache_data
def load_vocab():
    # 如果你有本地词表文件（csv）可以在这里加载并返回 dict(word->rank)
    # 尝试加载 data/coca_cleaned.csv 或 data.csv（与你原 app 的加载一致）
    possible = ["coca_cleaned.csv", "data.csv"]
    for f in possible:
        if os.path.exists(f):
            try:
                import pandas as pd
                df = pd.read_csv(f)
                cols = [str(c).strip().lower() for c in df.columns]
                df.columns = cols
                w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
                r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1] if len(cols)>1 else cols[0])
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
                df = df.sort_values(r_col, ascending=True).drop_duplicates(subset=[w_col], keep='first')
                return {row[w_col]: int(row[r_col]) for _, row in df.iterrows()}
            except Exception as e:
                print("加载本地词库失败:", e)
                break
    # fallback 示例词库（请替换）
    return {
        "abandon": 8000, "abstract": 8001, "academy": 8002, "accelerate": 8003,
        "accessory": 8004, "accommodate": 8005, "accompany": 8006, "accumulate": 8007,
        "accuracy": 8008, "acknowledge": 8009, "acquire": 8010
    }

vocab_dict = load_vocab()

# =====================================
# 词形归一函数（占位，可替换为你已有的 get_lemma）
# =====================================
def get_lemma(w: str) -> str:
    return w.lower()

# =====================================
# 词质量过滤 & 流式解析（低内存）
# =====================================
def is_valid_word(w: str) -> bool:
    if len(w) < 2 and w not in ("a", "i"):
        return False
    if w.count("'") > 1:
        return False
    return True

def stream_analyze_text(text: str):
    freq = Counter()
    tokens = []
    for chunk in text.split("\n"):
        words = re.findall(r"[a-zA-Z']+", chunk)
        for w in words:
            lemma = get_lemma(w)
            if not is_valid_word(lemma):
                continue
            freq[lemma] += 1
            tokens.append(lemma)
    return tokens, freq

# =====================================
# 高频短语检测（bigrams / trigrams）
# =====================================
def detect_phrases(tokens, min_freq=2):
    bigrams = Counter(zip(tokens, tokens[1:]))
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    phrases = []
    for gram, f in bigrams.items():
        if f >= min_freq:
            phrases.append((" ".join(gram), f))
    for gram, f in trigrams.items():
        if f >= min_freq:
            phrases.append((" ".join(gram), f))
    return phrases

# =====================================
# analyze_words：把 unique_words -> 带 rank/freq 的结构 (轻量)
# =====================================
def analyze_words(unique_words, freq_dict):
    rows = []
    for w in unique_words:
        # 如果是短语，用首词估计难度（保守做法）
        rank = vocab_dict.get(w.split()[0], 99999)
        rows.append({"word": w, "rank": rank, "freq": freq_dict.get(w, 1)})
    rows.sort(key=lambda r: (r["rank"], -r["freq"]))
    return rows

# =====================================
# UI：侧边栏（手机模式／简洁模式）
# =====================================
mobile_mode = st.sidebar.checkbox("📱 手机模式（简洁单列）", value=True)
min_phrase_freq = st.sidebar.slider("短语检测最小频次", 2, 10, 2)
top_n = st.sidebar.number_input("Top N 显示", 10, 1000, 100, step=10)
# 保存你常用 prompt 模板（可以扩展）
template_choice = st.sidebar.selectbox("Prompt 模板", ["Anki CSV（中英+例句）", "简洁例句（英文）", "自定义"])

# =====================================
# 主界面
# =====================================
st.markdown("### 1) 快速从文本筛选单词（支持粘贴 / 上传）")

col_left, col_right = st.columns([3,1]) if not mobile_mode else (st, None)

with col_left:
    raw_text = st.text_area("粘贴文本（或上传文件后自动填充）", height=180, key="raw_text")

st.markdown("---")

# 文件上传保持（保留文档解析功能），仅调用我们已有的 extract 方法或简单文本读取
uploaded = st.file_uploader("上传文档（txt/pdf/docx/epub）以提取文本（可选）", type=["txt","pdf","docx","epub"])
if uploaded is not None:
    try:
        ext = uploaded.name.split(".")[-1].lower()
        uploaded.seek(0)
        if ext == "txt":
            raw_text += "\n" + uploaded.getvalue().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            # 尝试用 PyPDF2 解析（若无库则忽略）
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded)
                txt = " ".join([p.extract_text() or "" for p in reader.pages])
                raw_text += "\n" + txt
            except Exception as e:
                st.warning("无法解析 PDF（服务器上未安装 PyPDF2），请粘贴文本或上传 txt。")
        elif ext == "docx":
            try:
                import docx
                doc = docx.Document(uploaded)
                raw_text += "\n" + "\n".join([p.text for p in doc.paragraphs])
            except:
                st.warning("不能解析 docx（未安装 python-docx）。")
        else:
            st.warning("不支持此文件类型的自动解析。")
    except Exception as e:
        st.error("文件解析出错：" + str(e))

# 分析按钮
if st.button("🔎 从文本筛词 & 检测短语"):
    if not raw_text.strip():
        st.warning("请先粘贴或上传文本")
        st.stop()
    t0 = time.time()
    tokens, freq_dict = stream_analyze_text(raw_text)
    phrases = detect_phrases(tokens, min_freq=min_phrase_freq)
    # 合并短语到词频表
    for p, f in phrases:
        freq_dict[p] += f
    unique_words = list(freq_dict.keys())
    rows = analyze_words(unique_words, freq_dict)
    dur = time.time() - t0
    st.success(f"完成：共检测到 {len(rows)} 个词/短语，用时 {dur:.2f}s")
    # 展示结果（可多选）
    st.markdown("#### 结果（可勾选要包含到 Prompt 的单词）")
    selected = []
    # 分页显示 top_n
    display_rows = rows[:top_n]
    for r in display_rows:
        chk = st.checkbox(f"{r['word']}  | Rank:{r['rank']}  | Freq:{r['freq']}", value=False, key=f"w_{r['word']}")
        if chk:
            selected.append(r["word"])
    # 如果用户没勾选，默认选 top 20
    if not selected:
        selected = [r["word"] for r in display_rows[:min(20, len(display_rows))]]
        st.info(f"未手动选择，默认使用前 {len(selected)} 个词生成 Prompt（可在生成后编辑）")

    # =====================================
    # 区间筛词与直接从词库按 rank 区间选词（你之前要的功能）
    # =====================================
    st.markdown("---")
    st.markdown("#### 或者：从词库中按词频区间选取单词（适合每日刷固定难度）")
    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        start_rank = st.number_input("起始 rank", min_value=1, max_value=20000, value=8000, step=1)
    with col_b:
        end_rank = st.number_input("结束 rank", min_value=1, max_value=20000, value=8020, step=1)
    with col_c:
        if st.button("📥 从词库筛选该区间"):
            chosen = [w for w,r in vocab_dict.items() if start_rank <= r <= end_rank]
            chosen.sort(key=lambda x: vocab_dict[x])
            if chosen:
                st.success(f"找到 {len(chosen)} 个单词（区间 {start_rank}-{end_rank}）")
                # 覆盖 selected 列表
                selected = chosen
            else:
                st.warning("该区间没有单词，请调整区间或加载完整词库")

    # =====================================
    # Prompt 生成（可编辑） & 导出
    # =====================================
    st.markdown("---")
    st.markdown("### 生成 AI Prompt（把下面的 prompt 复制到手机上的 ChatGPT / Gemini / Claude）")
    # 简单模板：可扩展为多个模板
    default_template = (
        "你是一个专业的 Anki 卡片生成器。\n\n"
        "请把以下单词或短语生成标准 CSV（两列：Front,Back），要求：\n"
        "- Front: 单词或短语\n"
        "- Back: 英文释义 + 中文翻译 + 1个英文例句（不要多余说明）\n"
        "- 输出纯 CSV，不要添加多余文字或解释\n\n"
        "单词列表（请逐个处理）:\n{words}\n"
    )
    auto_prompt = default_template.format(words=", ".join(selected))
    prompt_text = st.text_area("可编辑 Prompt（手机上复制粘贴到 AI）", auto_prompt, height=260)

    # 下载 prompt
    st.download_button("⬇️ 下载 Prompt 文本", data=prompt_text.encode("utf-8"), file_name=f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    st.info("提示：在 iPhone 上，你可以打开 ChatGPT App 或者 Safari 的 ChatGPT 页面，粘贴上述 Prompt，运行后得到 CSV 文本或文件。下面我也给了如何把 AI 返回的 CSV 导入 AnkiMobile 的说明（含操作步骤）。")

    # =====================================
    # 把 AI 的输出粘回应用（粘贴 CSV），并一键下载 CSV（供 Anki 导入）
    # =====================================
    st.markdown("---")
    st.markdown("### 如果 AI 返回了 CSV 文本 / 你已有 CSV：粘到下面并点击导出（适合直接导入 Anki）")
    pasted_csv = st.text_area("把 AI 返回的 CSV 文本粘在这里（或把 CSV 文件内容复制粘贴）", height=180)
    if st.button("📄 解析并导出为文件（可直接下载）"):
        if not pasted_csv.strip():
            st.warning("请先粘贴 CSV 文本（或从 AI 导出 CSV 并粘贴）")
        else:
            # 简单安全解析：按行分割，写到 csv 输出
            sio = io.StringIO()
            writer = csv.writer(sio)
            # 尝试智能解析：如果 AI 输出包含说明文字，尝试只取最像 CSV 的行（含逗号或制表符）
            lines = pasted_csv.strip().splitlines()
            good_lines = []
            for ln in lines:
                if "," in ln or "\t" in ln:
                    good_lines.append(ln)
            if not good_lines:
                # 把所有行当作单列 front/back 以制表符分隔
                for ln in lines:
                    writer.writerow([ln])
            else:
                for ln in good_lines:
                    # 试用 csv.reader 去解析每行
                    try:
                        rr = list(csv.reader([ln]))[0]
                        writer.writerow(rr)
                    except:
                        writer.writerow([ln])
            data = sio.getvalue().encode("utf-8-sig")
            fname = f"anki_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button("⬇️ 下载 CSV（含 BOM，适合 Anki 导入）", data=data, file_name=fname, mime="text/csv")

    # =====================================
    # 额外提示：如何把 CSV / APKG 导入到 iOS 上的 AnkiMobile
    # （官方说明和常用方法）
    # =====================================
    st.markdown("---")
    st.markdown("## iOS（AnkiMobile）导入指南（快速参考）")
    st.markdown("""
- 推荐方法：在电脑上用 Anki Desktop 导入 CSV 并导出 `.apkg`，然后通过 iCloud Drive / AirDrop / 文件（Files app）把 `.apkg` 传到手机并“Open in AnkiMobile”。  
- 直接在 iPhone 上：AnkiMobile 现在支持通过 Files / Open in 的方式导入 `.apkg` 或兼容的 CSV/text 文件；把刚刚下载的 CSV 文件保存到 `Files`（iCloud Drive），然后长按该文件 → 分享 → 选择 **AnkiMobile** 即可导入。参考 AnkiMobile 官方文档与讨论。 :contentReference[oaicite:0]{index=0}
- 另一种方便方案：将生成的 Prompt 发到手机（比如 Telegram / 邮件），在手机端运行 AI，获得 CSV 后保存到 Files，然后按上一步导入。  
- 如果你希望直接从手机将 `.apkg` 导入：可以把 `.apkg` 文件通过 AirDrop 发送到手机，接受后通常会提示用 AnkiMobile 打开并导入。 :contentReference[oaicite:1]{index=1}
""")

    st.markdown("---")
    st.success("流程说明：生成 Prompt → 在手机 AI 客户端粘贴并运行 → 得到 CSV / 文件 → 保存到 Files 或 AirDrop → 用 AnkiMobile 打开导入。")

# 页脚：提示与扩展
st.markdown("---")
st.caption("提示：本应用不再内置任何 AI 调用。它生成可复用的 Prompt，方便你在手机 AI 客户端（ChatGPT / Claude / Gemini）中生成最终卡片，并把卡片导入 AnkiMobile。")
