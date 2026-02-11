import streamlit as st
import re
import time
from collections import Counter
import io
import csv
from datetime import datetime

# =====================================
# 页面配置 + 手机友好样式
# =====================================

st.set_page_config(page_title="Vocab Master Mobile", layout="wide")

st.markdown("""
<style>
.block-container {
  max-width: 640px;
  margin: auto;
  padding: 12px;
}
.stButton>button {
  height: 48px;
  font-size: 16px;
}
textarea {
  font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Vocab Master — 手机版")

# =====================================
# 示例词库（替换为你的真实词库）
# =====================================

@st.cache_data
def load_vocab():
    return {
        "abandon": 8000,
        "abstract": 8001,
        "academy": 8002,
        "accelerate": 8003,
        "accessory": 8004,
        "accommodate": 8005,
        "accompany": 8006,
        "accumulate": 8007,
        "accuracy": 8008,
        "acknowledge": 8009,
        "acquire": 8010,
    }

vocab_dict = load_vocab()

# =====================================
# NLP 轻量函数
# =====================================

def get_lemma(w):
    return w.lower()

def is_valid_word(w):
    if len(w) < 2 and w not in ("a", "i"):
        return False
    if w.count("'") > 1:
        return False
    return True

def stream_analyze_text(text):
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

def analyze_words(unique_words, freq_dict):
    rows = []

    for w in unique_words:
        rank = vocab_dict.get(w.split()[0], 99999)

        rows.append({
            "word": w,
            "rank": rank,
            "freq": freq_dict.get(w, 1)
        })

    rows.sort(key=lambda r: (r["rank"], -r["freq"]))
    return rows

# =====================================
# UI 设置
# =====================================

mobile_mode = st.sidebar.checkbox("📱 手机模式", True)
min_phrase_freq = st.sidebar.slider("短语检测频率", 2, 10, 2)
top_n = st.sidebar.number_input("显示数量", 10, 500, 100)

# 修复布局 bug：container 必须始终可 with 使用
if mobile_mode:
    col_left = st.container()
else:
    col_left, _ = st.columns([3, 1])

# =====================================
# 文本筛词区域
# =====================================

with col_left:

    st.header("📥 文本筛词")

    raw_text = st.text_area("粘贴文本", height=180)

    if st.button("🔍 分析文本"):

        if not raw_text.strip():
            st.warning("请输入文本")
            st.stop()

        start = time.time()

        tokens, freq_dict = stream_analyze_text(raw_text)
        phrases = detect_phrases(tokens, min_phrase_freq)

        for p, f in phrases:
            freq_dict[p] += f

        rows = analyze_words(list(freq_dict.keys()), freq_dict)

        st.success(f"完成，用时 {time.time()-start:.2f}s")

        selected = []

        st.subheader("选择词汇")

        for r in rows[:top_n]:

            key = "w_" + str(abs(hash(r["word"])))

            if st.checkbox(
                f"{r['word']} | Rank:{r['rank']} | Freq:{r['freq']}",
                key=key
            ):
                selected.append(r["word"])

        if not selected:
            selected = [r["word"] for r in rows[:20]]

        # =====================================
        # 区间筛词
        # =====================================

        st.divider()
        st.subheader("🎯 词频区间筛选")

        c1, c2 = st.columns(2)

        with c1:
            start_rank = st.number_input("起始", 1, 20000, 8000)

        with c2:
            end_rank = st.number_input("结束", 1, 20000, 8010)

        if st.button("筛选区间单词"):

            selected = [
                w for w, r in vocab_dict.items()
                if start_rank <= r <= end_rank
            ]

            selected.sort(key=lambda w: vocab_dict[w])

            st.success(f"找到 {len(selected)} 个词")

        # =====================================
        # Prompt 生成
        # =====================================

        st.divider()
        st.subheader("🧠 AI Prompt")

        template = """你是一个专业 Anki 卡片生成器。

请把以下单词生成 CSV：

Front = 单词
Back = 英文释义 + 中文 + 例句

只输出 CSV：

{words}
"""

        prompt = template.format(words=", ".join(selected))

        st.text_area("复制到 AI 使用", prompt, height=200)

        st.download_button(
            "⬇ 下载 Prompt",
            prompt,
            file_name="prompt.txt"
        )

        # =====================================
        # CSV 导入区
        # =====================================

        st.divider()
        st.subheader("📄 AI 返回 CSV → 导出")

        pasted = st.text_area("粘贴 AI 返回 CSV", height=160)

        if st.button("导出 CSV"):

            if pasted.strip():

                sio = io.StringIO()
                writer = csv.writer(sio)

                for line in pasted.splitlines():
                    if "," in line:
                        writer.writerow(next(csv.reader([line])))

                data = sio.getvalue().encode("utf-8-sig")

                st.download_button(
                    "⬇ 下载 Anki CSV",
                    data,
                    file_name=f"anki_{datetime.now().strftime('%H%M%S')}.csv"
                )

# =====================================
# 使用说明
# =====================================

st.divider()

st.info("""
📱 iOS 使用流程：

1️⃣ 生成 Prompt → 复制到 ChatGPT / Claude  
2️⃣ AI 输出 CSV  
3️⃣ 粘贴回来 → 导出  
4️⃣ 在 iPhone 打开 CSV → 分享 → Anki 导入  

每天重复即可。
""")
