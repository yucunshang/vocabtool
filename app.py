import streamlit as st
import pandas as pd
import re
import os
import time
from datetime import datetime, timedelta, timezone
import lemminflect
import nltk
import genanki
import random
import tempfile
from bs4 import BeautifulSoup

# --- 文件处理库 ---
import pypdf
import docx
import ebooklib
from ebooklib import epub

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="Vocab Flow Ultra", page_icon="⚡️", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    .preview-table { font-size: 12px; }
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
    """加载词频表"""
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
# 2. 核心逻辑: 文本解析 (Smart Parsing)
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
    智能解析函数：处理 Markdown 表格、多余管道符、表头等问题
    """
    parsed_cards = []
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 1. 过滤 Markdown 分割线 (如 |---|---|)
        if set(line) == {'|', '-'} or '---' in line: continue
        
        # 2. 过滤表头
        if 'Word' in line and 'Definition' in line: continue

        # 3. 移除行首行尾的 Markdown 管道符 (重要!)
        # 很多时候 AI 会输出 "| word | ipa |"，如果不 strip('|')，split后第一个元素是空字符串，导致错位
        clean_line = line.strip('|')
        
        # 4. 分割
        parts = [p.strip() for p in clean_line.split('|')]
        
        # 5. 有效性检查 (至少要有单词和释义)
        if len(parts) >= 2:
            # 自动补全缺失字段，防止列表索引越界
            while len(parts) < 5:
                parts.append("")
                
            parsed_cards.append({
                'word': parts[0],
                'ipa': parts[1],
                'meaning': parts[2],
                'examples': parts[3],
                'etymology': parts[4]
            })
            
    return parsed_cards

# ==========================================
# 3. Anki 生成逻辑
# ==========================================
def generate_anki_package(cards_data, deck_name):
    # CSS 样式增强
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2f2f31; color: #f5f5f5; }
    .word { font-size: 40px; font-weight: bold; color: #007AFF; margin-bottom: 10px; }
    .nightMode .word { color: #5FA9FF; }
    .phonetic { color: #888; font-size: 18px; font-family: sans-serif; margin-bottom: 15px; }
    .def-container { text-align: left; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 15px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 22px; }
    .examples { background: #f4f4f4; padding: 15px; border-radius: 8px; color: #444; font-style: italic; font-size: 20px; line-height: 1.4; margin-bottom: 15px; }
    .etymology { display: block; font-size: 18px; color: #555; border: 1px dashed #bbb; padding: 8px 12px; border-radius: 6px; background-color: #fffaf0; margin-top: 10px; }
    """
    
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id, 
        f'VocabFlow Model {model_id}',
        fields=[{'name': 'Word'}, {'name': 'IPA'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="word">{{Word}}</div><div class="phonetic">{{IPA}}</div>',
            'afmt': '''{{FrontSide}}<div class="def-container"><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div><div class="etymology">🌱 <b>Etymology:</b> {{Etymology}}</div></div>''',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    
    for c in cards_data:
        deck.add_note(genanki.Note(
            model=model, 
            fields=[
                str(c['word']), str(c['ipa']), str(c['meaning']), 
                str(c['examples']).replace('\n','<br>'), str(c['etymology'])
            ]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words):
    w_list = ", ".join(words)
    return f"""
Act as a Dictionary API. Output strictly formatted data for Anki cards.

**Words:** {w_list}

**Strict Output Rules:**
1. **Format:** `Word | IPA | Definition | Examples | Etymology`
2. **NO Markdown Tables:** Do NOT output headers. Just the data lines.
3. **Separator:** Use `|` ONLY to separate fields.
4. **Mandatory Fields:**
   - Definition: Concise (<12 words).
   - Examples: 2 sentences, separated by `<br>`.
   - **Etymology:** REQUIRED. (e.g., "root(meaning)+suffix").

**Example:**
benevolent | /bəˈnevələnt/ | kind and helpful | He is **benevolent**.<br>A **benevolent** fund. | bene(good) + vol(wish)
"""

# ==========================================
# 4. 主程序 UI
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_extract, tab_anki = st.tabs(["1️⃣ 单词提取 & 生成", "2️⃣ 制作 Anki 牌组"])

# --- TAB 1: 提取 ---
with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析 (文本/文件)", "🔢 词频列表 (Rank & Random)"])
    
    with mode_context:
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略太简单的 (Current Level)", 1000, 20000, 4000, step=500)
        targ = c2.number_input("忽略太难的 (Target Level)", 2000, 50000, 15000, step=500)
        uploaded_file = st.file_uploader("📂 上传文档 (PDF/TXT/DOCX/EPUB)")
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
        gen_type = st.radio("生成模式", ["🔢 顺序截取", "🔀 范围随机 (Random)"], horizontal=True)
        if "顺序" in gen_type:
            c_a, c_b = st.columns(2)
            s_rank = c_a.number_input("起始排名", 1, 20000, 8000, step=100)
            count = c_b.number_input("数量", 10, 500, 50, step=10)
            if st.button("🚀 生成顺序列表"):
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
            r_count = c_cnt.number_input("Random Count", 10, 200, 50, step=10)
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
                    else: st.error("该范围无单词")

    if 'gen_words' in st.session_state and st.session_state['gen_words']:
        words = st.session_state['gen_words']
        st.divider()
        st.info(f"🎯 准备生成 **{len(words)}** 个单词")
        
        batch_size = st.number_input("AI 分组大小", 10, 200, 50)
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"第 {idx+1} 组 (单词 {idx*batch_size+1} - {idx*batch_size+len(batch)})", expanded=(idx==0)):
                st.code(get_ai_prompt(batch), language="markdown")

# --- TAB 2: Anki ---
with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    bj_time_str = get_beijing_time_str()
    
    if 'anki_input_text' not in st.session_state:
        st.session_state['anki_input_text'] = ""

    ai_resp = st.text_area(
        "在此粘贴 AI 的回复内容 (会自动修正格式错误)", 
        height=200, 
        key="anki_input_text"
    )
    
    deck_name = st.text_input("牌组名称", f"Vocab_{bj_time_str}")
    
    # --- 实时预览逻辑 ---
    if ai_resp.strip():
        # 调用智能解析
        parsed_data = parse_anki_data(ai_resp)
        
        if parsed_data:
            st.markdown("#### 👁️ 预览解析结果 (Verify before Download)")
            st.caption("请检查下表。如果表格列是对齐的，Anki 卡片就是正常的。")
            
            # 展示预览表格 (只取前几列用于展示)
            df_preview = pd.DataFrame(parsed_data)
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
            
            st.success(f"✅ 成功解析 {len(parsed_data)} 条数据")
            
            # 下载按钮只在数据有效时出现
            f_path = generate_anki_package(parsed_data, deck_name)
            with open(f_path, "rb") as f:
                st.download_button(
                    label=f"📥 下载 {deck_name}.apkg",
                    data=f,
                    file_name=f"{deck_name}.apkg",
                    mime="application/octet-stream",
                    type="primary"
                )
        else:
            st.warning("⚠️ 内容无法解析，请检查是否包含 '|' 分隔符")