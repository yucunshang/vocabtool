import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
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
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载
# ==========================================
@st.cache_resource(show_spinner="正在初始化 NLP 引擎...")
def load_nlp_resources():
    import nltk
    import lemminflect
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try: 
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError: 
                try: nltk.data.find(f'taggers/{pkg}') 
                except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        st.warning(f"NLP 资源加载部分异常 (不影响基础功能): {e}")
    return nltk, lemminflect

def get_file_parsers():
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

def get_genanki():
    import genanki
    import tempfile
    return genanki, tempfile

@st.cache_data
def load_vocab_data():
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            w_col = next((c for c in df.columns if 'word' in c), None)
            r_col = next((c for c in df.columns if 'rank' in c), None)
            
            if not w_col: w_col = df.columns[0]
            if not r_col: r_col = df.columns[1]

            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
        except Exception as e:
            st.error(f"词库加载失败: {e}")
            return {}, None
    return {}, None

VOCAB_DICT, FULL_DF = load_vocab_data()

def get_beijing_time_str():
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

# --- 修复点：安全的重置逻辑 ---
def clear_all_state():
    keys_to_drop = ['gen_words', 'raw_count', 'process_time']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    # 修复：通过改变 ID 来强制重置 uploader，而不是直接修改 key 的值
    st.session_state['uploader_reset_id'] = str(random.randint(1000, 9999))

# ==========================================
# 2. 核心逻辑
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            for encoding in ['utf-8', 'gb18030', 'latin-1', 'cp1252']:
                try:
                    text = bytes_data.decode(encoding)
                    break
                except: continue
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'epub':
            genanki, tempfile = get_genanki()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator=' ', strip=True) + " "
            try: os.remove(tmp_path)
            except: pass
    except Exception as e:
        return f"Error: {e}"
    return text

def is_valid_word(word):
    if len(word) < 2: return False
    if len(word) > 25: return False 
    if re.search(r'(.)\1{2,}', word): return False
    if not re.search(r'[aeiouy]', word): return False
    if re.search(r'\d', word): return False
    return True

def get_best_lemma_and_rank(word, vocab_dict, lemminflect):
    candidates = {}
    if word in vocab_dict:
        candidates[word] = vocab_dict[word]
    v_lemma = lemminflect.getLemma(word, upos='VERB')[0]
    if v_lemma in vocab_dict:
        candidates[v_lemma] = vocab_dict[v_lemma]
    n_lemma = lemminflect.getLemma(word, upos='NOUN')[0]
    if n_lemma in vocab_dict:
        candidates[n_lemma] = vocab_dict[n_lemma]

    if not candidates:
        return word, 99999
    
    best_lemma = min(candidates, key=candidates.get)
    return best_lemma, candidates[best_lemma]

def analyze_logic(text, current_lvl, target_lvl, include_unknown, mode="smart"):
    nltk, lemminflect = load_nlp_resources()
    text = re.sub(r'[’]', "'", text) 
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(raw_tokens)
    
    tokens = [t.lower() for t in raw_tokens if is_valid_word(t.lower())]
    unique_tokens = sorted(list(set(tokens)))
    
    final_results = []
    seen_lemmas = set()
    
    for w in unique_tokens:
        lemma_res, rank = get_best_lemma_and_rank(w, VOCAB_DICT, lemminflect)
        
        is_in_range = (rank >= current_lvl and rank <= target_lvl)
        is_unknown_included = (rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            if mode == "direct":
                final_results.append((w, rank))
            else:
                if lemma_res not in seen_lemmas:
                    final_results.append((lemma_res, rank))
                    seen_lemmas.add(lemma_res)
    
    final_results.sort(key=lambda x: x[1])
    return [x[0] for x in final_results], total_words

def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.replace("```json", "").replace("```", "").strip()
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases_lower = set()

    for match in matches:
        json_str = match.group()
        if json_str.endswith(",}"): json_str = json_str.replace(",}", "}")
        try:
            data = json.loads(json_str, strict=False)
            front_text = data.get("w", data.get("word", "")).strip()
            meaning = data.get("m", data.get("meaning", data.get("definition", ""))).strip()
            examples = data.get("e", data.get("examples", data.get("sentence", ""))).strip()
            etymology = data.get("r", data.get("etymology", data.get("root", ""))).strip()
            
            if not etymology or etymology.lower() in ["none", "null", ""]:
                etymology = ""

            if not front_text or not meaning: continue
            
            front_text = front_text.replace('**', '')
            if front_text.lower() in seen_phrases_lower: continue
            seen_phrases_lower.add(front_text.lower())

            parsed_cards.append({
                'front_phrase': front_text,
                'meaning': meaning,
                'examples': examples,
                'etymology': etymology
            })
        except: continue
    return parsed_cards

def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
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
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id, f'VocabFlow JSON Model {model_id}',
        fields=[{'name': 'FrontPhrase'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{FrontPhrase}}</div>', 
            'afmt': '{{FrontSide}}<hr><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div>{{#Etymology}}<div class="footer-info"><div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div></div>{{/Etymology}}',
        }], css=CSS
    )
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[str(c['front_phrase']), str(c['meaning']), str(c['examples']).replace('\n','<br>'), str(c['etymology'])]))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    w_instr = "Key `w`: The word itself (lowercase)." if front_mode == "单词 (Word)" else "Key `w`: A short practical collocation/phrase (2-5 words) containing the word."
    m_instr = "Key `m`: Concise Chinese definition (max 15 chars)." if def_mode == "中文" else ("Key `m`: English Definition + Chinese Definition." if def_mode == "中英双语" else "Key `m`: English definition (concise).")
    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."
    r_instr = "Key `r`: Simplified Chinese Etymology (Root/Prefix) explaining why the word has this meaning." if need_ety else "Key `r`: Leave this empty string \"\"."

    return f"""
Role: Anki Card Generator.
Task: Create high-quality vocabulary cards for English learners.
Target Words: {w_list}

**OUTPUT FORMAT: NDJSON (One valid JSON object per line).**
No markdown, no lists, just JSON objects.

**Field Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**JSON Keys:** `w` (Front), `m` (Meaning), `e` (Examples), `r` (Etymology)
**Start generating now:**
"""

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失词库文件 (`coca_cleaned.csv`)，无法进行筛选。")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

with tab_guide:
    st.markdown("""
    ### 👋 欢迎使用 Vocab Flow Ultra
    <div class="guide-step"><span class="guide-title">Step 1: 提取生词</span> 上传文件或粘贴文本，系统自动过滤。</div>
    <div class="guide-step"><span class="guide-title">Step 2: 获取 Prompt</span> 复制 Prompt 给 AI。</div>
    <div class="guide-step"><span class="guide-title">Step 3: 制作 Anki</span> 粘贴 JSON 并下载。</div>
    """, unsafe_allow_html=True)

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析", "🔢 词频列表"])
    
    with mode_context:
        st.write("🛠️ **处理模式**")
        proc_mode = st.radio("选择模式", ["📖 智能分析 (文章/小说)", "📋 直通模式 (单词表/生词本)"], horizontal=True, label_visibility="collapsed")
        is_smart_mode = ("智能" in proc_mode)
        
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略排名前 N 的词", 1, 20000, 2000, step=100)
        targ = c2.number_input("忽略排名后 N 的词", 2000, 50000, 20000, step=500)
        include_unknown = st.checkbox("🔓 包含未收录词 (人名/生僻词)", value=False)
        
        # --- 修复点：使用动态Key ---
        if 'uploader_reset_id' not in st.session_state:
            st.session_state['uploader_reset_id'] = "1"
            
        uploader_key = f"uploader_{st.session_state['uploader_reset_id']}"
        uploaded_file = st.file_uploader("📂 上传文档 (TXT/PDF/DOCX/EPUB)", key=uploader_key)
        
        pasted_text = st.text_area("📄 ...或粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 读取文件并清洗...")
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 2:
                    status.write("🔍 智能匹配词频...")
                    mode_str = "smart" if is_smart_mode else "direct"
                    final_words, raw_count = analyze_logic(raw_text, curr, targ, include_unknown, mode=mode_str)
                    
                    st.session_state['gen_words'] = final_words
                    st.session_state['raw_count'] = raw_count
                    st.session_state['process_time'] = time.time() - start_time
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容太短", state="error")
        
        if st.button("🗑️ 清空结果", type="secondary", on_click=clear_all_state): pass

    with mode_rank:
        st.info("此功能可直接从词库中批量抽取单词。")
        gen_type = st.radio("模式", ["🔢 顺序抽取", "🔀 随机抽取"], horizontal=True)
        if "顺序" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
             count = c_b.number_input("数量", 10, 500, 50, step=10)
             if st.button("🚀 生成列表"):
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                     st.session_state['gen_words'] = subset[w_col].tolist()
                     st.session_state['raw_count'] = 0
                     st.session_state['process_time'] = 0
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("Min Rank", 1, 20000, 1, step=100)
             max_r = c_max.number_input("Max Rank", 1, 25000, 5000, step=100)
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
                         st.session_state['raw_count'] = 0
                         st.session_state['process_time'] = 0

    if 'gen_words' in st.session_state and st.session_state['gen_words']:
        words = st.session_state['gen_words']
        st.divider()
        st.markdown("### 📊 分析报告")
        k1, k2, k3 = st.columns(3)
        raw_c = st.session_state.get('raw_count', 0)
        p_time = st.session_state.get('process_time', 0.1)
        k1.metric("📄 文档总字数", f"{raw_c:,}")
        k2.metric("🎯 筛选生词", f"{len(words)}")
        k3.metric("⚡ 耗时", f"{p_time:.2f}s")
        
        st.markdown("### 📋 单词预览")
        st.code(", ".join(words), language="text")

        with st.expander("⚙️ **Prompt 设置**", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("正面内容", ["短语搭配 (Phrase)", "单词 (Word)"])
            def_mode = col_s2.selectbox("背面释义", ["英文", "中文", "中英双语"])
            col_s3, col_s4 = st.columns(2)
            ex_count = col_s3.slider("例句数量", 1, 3, 1)
            need_ety = col_s4.checkbox("包含词源", value=True)

        batch_size = st.number_input("AI 分组大小", 10, 200, 50, step=10)
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 第 {idx+1} 组 (共 {len(batch)} 词)", expanded=(idx==0)):
                prompt_text = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.text_area(f"prompt_area_{idx}", value=prompt_text, height=150)

with tab_anki:
    st.markdown("### 📦 制作 Anki")
    bj_time_str = get_beijing_time_str()
    st.caption("👇 粘贴 AI 回复：")
    ai_resp = st.text_area("JSON 输入框", height=300, key="anki_input_text")
    deck_name = st.text_input("牌组名", f"Vocab_{bj_time_str}")
    
    if ai_resp.strip():
        parsed_data = parse_anki_data(ai_resp)
        if parsed_data:
            st.success(f"✅ 成功解析 {len(parsed_data)} 条数据")
            df_view = pd.DataFrame(parsed_data)
            df_view.rename(columns={'front_phrase': '正面', 'meaning': '背面', 'etymology': '词源'}, inplace=True)
            st.dataframe(df_view[['正面', '背面', '词源']], use_container_width=True, hide_index=True)
            
            f_path = generate_anki_package(parsed_data, deck_name)
            with open(f_path, "rb") as f:
                st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
        else:
            st.warning("⚠️ 解析失败，请检查是否为 JSON 格式。")