import streamlit as st
import pandas as pd
import re
import os
import random
import tempfile
import lemminflect
import nltk
import genanki
from bs4 import BeautifulSoup

# --- 文件处理库 ---
import pypdf
import docx
import ebooklib
from ebooklib import epub
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载
# ==========================================
@st.cache_resource
def setup_nltk():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try: nltk.data.find(f'tokenizers/{pkg}')
            except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except: pass
setup_nltk()

@st.cache_data
def load_vocab_data():
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
        except: return {}, None
    return {}, None

VOCAB_DICT, FULL_DF = load_vocab_data()

def get_lemma(word):
    try: return lemminflect.getLemma(word, upos='VERB')[0]
    except: return word

def get_beijing_time_str():
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

def clear_all_state():
    st.session_state.clear()

# ==========================================
# 2. 核心解析逻辑 (V7 无损解析版)
# ==========================================
def extract_text_from_file(uploaded_file):
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'epub':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + " "
            os.remove(tmp_path)
    except Exception as e:
        return f"Error: {e}"
    return text

def analyze_logic(text, current_lvl, target_lvl):
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    total_words = len(raw_tokens)
    unique_tokens = set(raw_tokens)
    target_words = []
    for w in unique_tokens:
        if len(w) < 3: continue 
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999)
        if rank > current_lvl and rank <= target_lvl:
            target_words.append((lemma, rank))
    target_words.sort(key=lambda x: x[1])
    return [x[0] for x in target_words], total_words

def parse_anki_data(raw_text):
    """
    V7 解析器：
    不使用 filter，而是使用 pop 移除首尾空管，确保保留所有中间列。
    """
    parsed_cards = []
    # 统一全角符号
    raw_text = raw_text.replace('｜', '|')
    lines = raw_text.strip().split('\n')
    seen_phrases = set()

    for line in lines:
        line = line.strip()
        if not line: continue
        if '---' in line: continue
        if 'Phrase' in line and 'Definition' in line: continue

        # 1. 原始分割
        segments = line.split('|')
        
        # 2. 智能去除 Markdown 表格两侧的空字符串
        # 例如: "| A | B | C |" -> ["", "A", "B", "C", ""]
        # 我们只移除首尾的空串，保留中间的空串(防止错位)
        if len(segments) > 1 and segments[0].strip() == "":
            segments.pop(0)
        if len(segments) > 0 and segments[-1].strip() == "":
            segments.pop(-1)
            
        # 3. 清洗每一列的内容
        parts = [s.strip() for s in segments]
        
        # 4. 提取数据 (允许容错，最少3列)
        if len(parts) >= 3:
            front_text = parts[0]
            
            # --- 清洗正面 ---
            front_text = front_text.rstrip('.,?!:; ')
            front_text = front_text.replace('*', '')
            
            # 过滤超长句子 (>7单词)
            if len(front_text.split()) > 7:
                continue

            # 首字母小写
            if front_text:
                first_word = front_text.split()[0]
                if first_word != "I" and not first_word.isupper():
                    front_text = front_text[0].lower() + front_text[1:]

            if front_text in seen_phrases:
                continue
            seen_phrases.add(front_text)

            meaning = parts[1]
            examples = parts[2]
            
            # --- 关键修复：获取词源 ---
            # 如果有第4列，取第4列；否则提示缺失
            etymology = parts[3] if len(parts) >= 4 else "⚠️ 缺失：AI未生成此列"

            parsed_cards.append({
                'front_phrase': front_text,
                'meaning': meaning,
                'examples': examples,
                'etymology': etymology
            })
            
    return parsed_cards

# ==========================================
# 3. Anki 生成逻辑 (无 IPA, 包含词源)
# ==========================================
def generate_anki_package(cards_data, deck_name):
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; line-height: 1.3; }
    .nightMode .phrase { color: #66b0ff; }
    
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .nightMode .definition { color: #e0e0e0; }
    
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 18px; line-height: 1.5; margin-bottom: 15px; text-align: left; }
    .nightMode .examples { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    
    .footer-info { margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 10px; text-align: left; }
    
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; line-height: 1.4; border: 1px solid #fef3c7; }
    .etymology b { color: #8b5cf6; } 
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    .nightMode .etymology b { color: #a78bfa; }
    """
    
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id, 
        f'VocabFlow NoIPA Model {model_id}',
        fields=[
            {'name': 'FrontPhrase'}, 
            {'name': 'Meaning'}, 
            {'name': 'Examples'}, 
            {'name': 'Etymology'}
        ],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{FrontPhrase}}</div>', 
            'afmt': '''
            {{FrontSide}}
            <hr>
            <div class="definition">{{Meaning}}</div>
            <div class="examples">{{Examples}}</div>
            
            <div class="footer-info">
                <div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div>
            </div>
            ''',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(
            model=model, 
            fields=[
                str(c['front_phrase']), 
                str(c['meaning']), 
                str(c['examples']).replace('\n','<br>'), 
                str(c['etymology'])
            ]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt 生成逻辑 (V7 终极版)
# ==========================================
def get_ai_prompt(words):
    w_list = ", ".join(words)
    return f"""
Task: Create Anki cards.
Words: {w_list}

**STRICT OUTPUT FORMAT (4 Columns):**
`Phrase | English Definition | Example Sentences | Chinese Etymology`

**RULES:**
1. **NO IPA.** 
2. **Column 1 (Phrase):** Short collocation (2-5 words). NO sentences. Lowercase.
3. **Column 3 (Examples):** 1-2 sentences. Use `<br>` to separate.
4. **Column 4 (Etymology):** Simplified Chinese only. **MUST NOT BE EMPTY.** If unknown, write "词源暂缺".

**Example:**
a benevolent leader | characterized by goodwill | The benevolent man helped the poor.<br>She is benevolent. | 词根: bene (好) + vol (意愿)

**Start Output:**
"""

# ==========================================
# 5. 主程序 UI
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_extract, tab_anki = st.tabs(["1️⃣ 单词提取 & 生成", "2️⃣ 制作 Anki 牌组"])

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析", "🔢 词频列表"])
    
    with mode_context:
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略太简单的 (Current Level)", 1000, 20000, 4000, step=500)
        targ = c2.number_input("忽略太难的 (Target Level)", 2000, 50000, 15000, step=500)
        uploaded_file = st.file_uploader("📂 上传文档 (支持多种格式)")
        pasted_text = st.text_area("📄 ...或粘贴文本", height=100)
        
        if st.button("🚀 开始分析", type="primary"):
            raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
            if len(raw_text) > 10:
                final_words, total = analyze_logic(raw_text, curr, targ)
                st.session_state['gen_words'] = final_words
                st.session_state['total_count'] = total
            else: st.warning("内容无效")
        if st.button("🗑️ 清空所有数据", type="secondary", on_click=clear_all_state): pass

    with mode_rank:
        gen_type = st.radio("生成模式", ["🔢 顺序截取", "🔀 范围随机"], horizontal=True)
        if "顺序" in gen_type:
            c_a, c_b = st.columns(2)
            s_rank = c_a.number_input("起始排名", 1, 20000, 8000, step=100)
            count = c_b.number_input("数量", 10, 500, 50, step=10)
            if st.button("🚀 生成列表"):
                if FULL_DF is not None:
                    r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                    w_col = next(c for c in FULL_DF.columns if 'word' in c)
                    subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                    st.session_state['gen_words'] = subset[w_col].tolist()
                    st.session_state['total_count'] = count
        else:
            c_min, c_max, c_cnt = st.columns([1,1,1])
            min_r = c_min.number_input("Min Rank", 1, 20000, 6000, step=500)
            max_r = c_max.number_input("Max Rank", 1, 25000, 8000, step=500)
            r_count = c_cnt.number_input("Count", 10, 200, 50, step=10)
            if st.button("🎲 随机抽取"):
                if FULL_DF is not None:
                    r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                    w_col = next(c for c in FULL_DF.columns if 'word' in c)
                    mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                    candidates = FULL_DF[mask]
                    if len(candidates) > 0:
                        subset = candidates.sample(n=min(r_count, len(candidates))).sort_values(r_col)
                        st.session_state['gen_words'] = subset[w_col].tolist()
                        st.session_state['total_count'] = len(subset)
                        st.success(f"抽取了 {len(subset)} 个单词")

    if 'gen_words' in st.session_state and st.session_state['gen_words']:
        words = st.session_state['gen_words']
        st.divider()
        st.info(f"🎯 待处理单词: {len(words)} 个")
        
        batch_size = st.number_input("AI 分组大小", 10, 200, 50, step=10)
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"第 {idx+1} 组 (复制发给 AI)", expanded=(idx==0)):
                st.code(get_ai_prompt(batch), language="markdown")

with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    bj_time_str = get_beijing_time_str()
    if 'anki_input_text' not in st.session_state: st.session_state['anki_input_text'] = ""

    ai_resp = st.text_area("在此粘贴 AI 回复", height=200, key="anki_input_text")
    deck_name = st.text_input("牌组名称", f"Vocab_{bj_time_str}")
    
    if ai_resp.strip():
        parsed_data = parse_anki_data(ai_resp)
        if parsed_data:
            st.markdown(f"#### 👁️ 预览 (成功解析 {len(parsed_data)} 条)")
            df_view = pd.DataFrame(parsed_data)
            df_view.rename(columns={'front_phrase': '正面 (短语)', 'meaning': '英文释义', 'examples': '例句', 'etymology': '中文词源'}, inplace=True)
            st.dataframe(df_view, use_container_width=True, hide_index=True)
            
            f_path = generate_anki_package(parsed_data, deck_name)
            with open(f_path, "rb") as f:
                st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
        else:
            st.warning("⚠️ 格式解析失败，请检查 AI 内容。")