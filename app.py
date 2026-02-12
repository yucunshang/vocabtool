import streamlit as st
import pandas as pd
import re
import os
import io
import time
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
# 0. 页面基础配置 & 样式
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
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; }
    .batch-container { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom: 10px; background-color: #f9f9f9; }
    .stat-box { padding: 10px; background-color: #e6fffa; border-radius: 8px; text-align: center; color: #006d5b; margin-bottom: 10px; }
    .reset-btn { color: red; border-color: red; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载 & 工具函数
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

def clear_all_state():
    """一键清空的回调函数"""
    st.session_state.clear()

# ==========================================
# 2. 文本提取与分析
# ==========================================
def extract_text_from_file(uploaded_file):
    """多格式文件解析"""
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
    """核心分析逻辑"""
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    total_words = len(raw_tokens)
    unique_tokens = set(raw_tokens)
    
    target_words = []
    for w in unique_tokens:
        if len(w) < 3: continue 
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999)
        
        # 筛选: Current < Rank <= Target
        if rank > current_lvl and rank <= target_lvl:
            target_words.append((lemma, rank))
            
    target_words.sort(key=lambda x: x[1])
    final_list = [x[0] for x in target_words]
    return final_list, total_words

# ==========================================
# 3. Anki 生成逻辑
# ==========================================
def generate_anki_package(cards_data, deck_name="Vocab_Deck"):
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2f2f31; color: #f5f5f5; }
    .word { font-size: 40px; font-weight: bold; color: #007AFF; margin-bottom: 10px; }
    .nightMode .word { color: #5FA9FF; }
    .phonetic { color: #888; font-size: 18px; font-family: sans-serif; }
    .def-container { text-align: left; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 10px; }
    .definition { font-weight: bold; color: #444; margin-bottom: 10px; }
    .nightMode .definition { color: #ddd; }
    .examples { background: #f4f4f4; padding: 10px; border-radius: 5px; color: #555; font-style: italic; font-size: 16px; }
    .nightMode .examples { background: #333; color: #ccc; }
    .etymology { margin-top: 15px; font-size: 14px; color: #888; border: 1px dashed #ccc; padding: 5px; display: inline-block;}
    """
    
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31), 'VocabFlow Model',
        fields=[{'name': 'Word'}, {'name': 'IPA'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="word">{{Word}}</div><div class="phonetic">{{IPA}}</div>',
            'afmt': '{{FrontSide}}<div class="def-container"><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div><div class="etymology">🌱 {{Etymology}}</div></div>',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[c['word'], c['ipa'], c['meaning'], c['examples'].replace('\n','<br>'), c['etymology']]))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words):
    w_list = ", ".join(words)
    return f"""
Act as a Lexicographer. Create Anki card data.
Words: {w_list}

**Strict Output Format (Pipe Separated `|`, NO Header):**
Word | IPA | Concise English Definition | 2 English Sentences | Etymology (Root+Suffix)

**Requirements:**
1. Definition: Simple English (B2 level).
2. Examples: 2 sentences separated by `<br>`.
3. Etymology: Format `root(meaning) + suffix`.
4. NO Header row.

**Example:**
benevolent | /bəˈnevələnt/ | kind and meaningful | He is benevolent.<br>A benevolent fund. | bene(good) + vol(wish)
"""

# ==========================================
# 4. 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

# 侧边栏：一键重置
with st.sidebar:
    st.header("控制台")
    if st.button("🗑️ 清空所有数据", type="secondary", on_click=clear_all_state):
        pass # 回调已处理

# Input Tabs
tab_input, tab_anki = st.tabs(["1️⃣ 提取 & 生成", "2️⃣ 打包 Anki"])

with tab_input:
    # 1. 来源选择
    input_method = st.radio("选择来源", ["📄 粘贴文本", "📂 上传文件", "🔢 词频Rank生成"], horizontal=True, label_visibility="collapsed")
    
    final_words = []
    
    # --- A. 文本/文件逻辑 ---
    if input_method in ["📄 粘贴文本", "📂 上传文件"]:
        c1, c2 = st.columns(2)
        curr = c1.number_input("Current Level (Ignore <)", 1000, 20000, 4000, step=500)
        targ = c2.number_input("Target Level (Ignore >)", 2000, 50000, 15000, step=500)
        
        raw_text = ""
        
        if input_method == "📄 粘贴文本":
            raw_text = st.text_area("在此粘贴文本", height=200)
            if st.button("🔍 分析文本"):
                if raw_text:
                    final_words, total = analyze_logic(raw_text, curr, targ)
                    st.session_state['gen_words'] = final_words
                    st.session_state['total_count'] = total
        else:
            up_file = st.file_uploader("支持 PDF/TXT/DOCX/EPUB", type=['txt','pdf','docx','epub'])
            if up_file and st.button("🚀 分析文件"):
                with st.spinner("解析中..."):
                    raw_text = extract_text_from_file(up_file)
                    if len(raw_text) > 10:
                        final_words, total = analyze_logic(raw_text, curr, targ)
                        st.session_state['gen_words'] = final_words
                        st.session_state['total_count'] = total
                    else:
                        st.error("无法读取文件内容")

    # --- B. 词频生成逻辑 ---
    else:
        c_a, c_b = st.columns(2)
        s_rank = c_a.number_input("Start Rank", 1, 20000, 8000, step=100)
        count = c_b.number_input("Count", 10, 500, 50, step=10)
        if st.button("🔢 生成列表"):
            if FULL_DF is not None:
                try:
                    r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                    w_col = next(c for c in FULL_DF.columns if 'word' in c)
                    subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                    st.session_state['gen_words'] = subset[w_col].tolist()
                    st.session_state['total_count'] = count
                except: st.error("数据源格式错误")

    # --- 结果展示 & 分批 Prompt ---
    if 'gen_words' in st.session_state:
        words = st.session_state['gen_words']
        
        st.divider()
        st.markdown(f"""
        <div class="stat-box">
            📊 来源词数: {st.session_state.get('total_count', 0)} | 
            🎯 筛选结果: <b>{len(words)}</b> 个单词
        </div>
        """, unsafe_allow_html=True)

        if len(words) > 0:
            # 分批设置
            c_batch, c_info = st.columns([1, 3])
            batch_size = c_batch.number_input("每组单词数 (Batch Size)", 10, 100, 30, step=10)
            c_info.info(f"💡 单词较多时，AI 容易输出中断。建议每组 20-40 个。共需 {len(words)//batch_size + (1 if len(words)%batch_size else 0)} 次生成。")
            
            # 自动分批逻辑
            batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
            
            st.markdown("### 🤖 AI Prompt 生成区 (分批)")
            
            for idx, batch in enumerate(batches):
                with st.expander(f"第 {idx+1} 组 (单词 {idx*batch_size+1} - {idx*batch_size+len(batch)})", expanded=(idx==0)):
                    st.write(f"包含: {', '.join(batch[:5])}...")
                    
                    # 生成该批次的 prompt
                    prompt = get_ai_prompt(batch)
                    st.code(prompt, language="markdown")
                    st.caption("👆 点击右上角复制，发给 AI。完成后复制下一组。")

with tab_anki:
    st.markdown("### 📦 打包 Anki (.apkg)")
    st.caption("在此处粘贴 AI 回复的所有内容。你可以把多次生成的回复粘贴在一起（换行分隔）。")
    
    ai_resp = st.text_area("粘贴内容 (支持多次粘贴)", height=300, placeholder="word1 | ...\nword2 | ...")
    deck_name = st.text_input("牌组命名", "VocabFlow Deck")
    
    if st.button("🚀 生成 .apkg 文件", type="primary"):
        if not ai_resp.strip():
            st.error("内容为空")
        else:
            cards = []
            skipped = 0
            # 宽容解析：过滤空行和可能的表头
            for line in ai_resp.strip().split('\n'):
                line = line.strip()
                if not line: continue
                if "|" not in line: continue
                if "Word | IPA" in line or "---" in line: continue 
                
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    cards.append({
                        'word': parts[0],
                        'ipa': parts[1] if len(parts) > 1 else '',
                        'meaning': parts[2] if len(parts) > 2 else '',
                        'examples': parts[3] if len(parts) > 3 else '',
                        'etymology': parts[4] if len(parts) > 4 else ''
                    })
                else:
                    skipped += 1
            
            if cards:
                f_path = generate_anki_package(cards, deck_name)
                with open(f_path, "rb") as f:
                    st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
                
                st.success(f"成功打包 {len(cards)} 张卡片！")
                if skipped > 0:
                    st.warning(f"跳过了 {skipped} 行格式不符的数据")
            else:
                st.error("未找到有效数据，请检查是否使用了 | 分隔符")