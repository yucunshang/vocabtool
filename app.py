import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import json
import time
import requests
import zipfile
import concurrent.futures
from collections import Counter
from io import BytesIO

# ==========================================
# 1. 基础配置与增强型 CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Fira Code', 'Consolas', monospace !important; font-size: 15px !important; }
    .main .block-container { padding-top: 2rem; }
    .stMetric { background: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #d1d5db; }
    .param-box { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 25px; border-left: 5px solid #ff4b4b; }
    .copy-hint { color: #6b7280; font-size: 0.85rem; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心 NLP 与 数据逻辑
# ==========================================

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    """带容错的知识库加载"""
    base_path = "data"
    default_res = ({}, {}, {}, set())
    try:
        def load_json(name):
            p = os.path.join(base_path, name)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: return json.load(f)
            return {}
        terms = {k.lower(): v for k, v in load_json('terms.json').items()}
        proper = {k.lower(): v for k, v in load_json('proper.json').items()}
        patch = load_json('patch.json')
        ambiguous = set(load_json('ambiguous.json'))
        return terms, proper, patch, ambiguous
    except Exception:
        return default_res

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource
def setup_nltk():
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    for pkg in ['averaged_perceptron_tagger', 'punkt']:
        try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except: pass
setup_nltk()

LEMMA_CACHE = {}
def get_lemma_optimized(w: str) -> str:
    w_lower = w.lower()
    if w_lower in LEMMA_CACHE: return LEMMA_CACHE[w_lower]
    lemmas_dict = lemminflect.getAllLemmas(w_lower)
    if not lemmas_dict: res = w_lower
    else:
        res = w_lower
        for pos in ['VERB', 'ADJ', 'NOUN', 'ADV']:
            if pos in lemmas_dict:
                res = lemmas_dict[pos][0]
                break
    LEMMA_CACHE[w_lower] = res
    return res

@st.cache_data
def load_vocab():
    vocab = {}
    for f_name in ["coca_cleaned.csv", "data.csv", "data/coca.csv"]:
        if os.path.exists(f_name):
            try:
                df = pd.read_csv(f_name)
                df.columns = [str(c).strip().lower() for c in df.columns]
                w_col = next((c for c in df.columns if 'word' in c or '单词' in c), df.columns[0])
                r_col = next((c for c in df.columns if 'rank' in c or '排序' in c), df.columns[1])
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
                df = df.sort_values(r_col).drop_duplicates(subset=[w_col])
                vocab = dict(zip(df[w_col], df[r_col]))
                break
            except: continue
    vocab.update(BUILTIN_PATCH_VOCAB)
    vocab.update({"china": 400, "google": 800, "apple": 800, "monday": 300, "january": 400})
    return vocab

VOCAB_DICT = load_vocab()

# ==========================================
# 3. 文档处理与 Prompt 引擎
# ==========================================

def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        content = uploaded_file.read()
        if ext == 'txt': return content.decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            import PyPDF2
            reader = PyPDF2.PdfReader(BytesIO(content))
            return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == 'docx':
            import docx
            doc = docx.Document(BytesIO(content))
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == 'epub':
            with zipfile.ZipFile(BytesIO(content)) as z:
                return " ".join([re.sub(r'<[^>]+>', ' ', z.read(f).decode('utf-8', errors='ignore')) 
                                for f in z.namelist() if f.endswith(('.html', '.xhtml'))])
    except Exception as e:
        st.error(f"解析失败: {e}")
    return ""

def get_base_prompt_template(export_format="TXT"):
    return f"""【角色】你是一位 Anki 闪卡专家。请处理以下单词列表：
1. 原子性：每个义项独立成卡。
2. 正面：自然的短语或搭配。
3. 背面：HTML排版，包含 [英文释义]<br><br><em>[例句]</em><br><br>【词根词缀】[中文解析]。
4. 格式：{export_format}，每个字段用双引号包裹，逗号分隔。
不要输出任何 Markdown 语法标记（如 ```csv），直接输出纯文本内容。"""

# ==========================================
# 4. AI 并发引擎
# ==========================================

def _fetch_deepseek_chunk(batch_words, prompt_template, api_key):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    full_prompt = f"{prompt_template}\n\n待处理词：\n{', '.join(batch_words)}"
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": full_prompt}], "temperature": 0.3}
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 429: time.sleep(5); continue
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'].strip()
            return re.sub(r'^```[a-zA-Z]*\n|\n```$', '', content)
        except Exception as e:
            if attempt == 2: return f"❌ 错误: {e}"
            time.sleep(2)
    return "❌ 超时"

def call_deepseek_api_chunked(prompt_template, words, progress_bar, status_container):
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return "⚠️ 未配置 API KEY"
    
    CHUNK_SIZE = 25
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    results = [None] * len(chunks)
    
    with st.status("🚀 AI 并发引擎处理中...", expanded=True) as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {executor.submit(_fetch_deepseek_chunk, chunks[i], prompt_template, api_key): i for i in range(len(chunks))}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                progress_bar.progress(completed / len(chunks))
                status.write(f"✅ 进度: {completed}/{len(chunks)} 批次")
        status.update(label="✨ 生成完毕", state="complete", expanded=False)
    return "\n".join(filter(None, results))

# ==========================================
# 5. 分析流水线
# ==========================================

def process_pipeline(text):
    raw_words = re.findall(r"\b[a-zA-Z']{2,}\b", text)
    if not raw_words: return None, None
    word_counts = Counter(raw_words)
    unique_lemmas_map = {}
    for word, count in word_counts.items():
        lemma = get_lemma_optimized(word)
        unique_lemmas_map[lemma] = unique_lemmas_map.get(lemma, 0) + count
    
    data = []
    for lemma, count in unique_lemmas_map.items():
        rank = VOCAB_DICT.get(lemma, 99999)
        display = PROPER_NOUNS_DB.get(lemma, lemma)
        if lemma in BUILTIN_TECHNICAL_TERMS: display = f"{lemma} ({BUILTIN_TECHNICAL_TERMS[lemma]})"
        data.append({"word": display, "rank": rank, "count": count, "raw": lemma})
    return pd.DataFrame(data).sort_values('rank'), raw_words

# ==========================================
# 6. UI 界面
# ==========================================

st.title("🚀 Vocab Master Pro")

if "is_processed" not in st.session_state: st.session_state.is_processed = False

# 参数区
with st.container():
    st.markdown("<div class='param-box'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: cur_lv = st.number_input("🎯 掌握词频起", 0, 20000, 3500)
    with c2: tgt_lv = st.number_input("🎯 目标词频止", 0, 30000, 12000)
    with c3: top_n = st.number_input("🔥 精选 Top N", 5, 500, 50)
    with c4: show_rank = st.checkbox("显示词频", True)
    st.markdown("</div>", unsafe_allow_html=True)

# 输入区
col_in1, col_in2 = st.columns([2, 1])
with col_in1: raw_input = st.text_area("📥 粘贴文本", height=150)
with col_in2: uploaded_file = st.file_uploader("📂 上传文档", type=["txt", "pdf", "docx", "epub"])

if st.button("🚀 极速解析", type="primary", use_container_width=True):
    content = raw_input + ("\n" + extract_text_from_file(uploaded_file) if uploaded_file else "")
    if content.strip():
        with st.spinner("分析中..."):
            df, raw_words = process_pipeline(content)
            if df is not None:
                st.session_state.df = df
                st.session_state.raw_count = len(raw_words)
                st.session_state.is_processed = True
    else: st.warning("内容为空")

# 渲染逻辑
if st.session_state.get("is_processed"):
    df = st.session_state.df
    m1, m2, m3 = st.columns(3)
    m1.metric("词汇总量", st.session_state.raw_count)
    m2.metric("独立词根", len(df))
    m3.metric("需重点学", len(df[(df['rank'] > cur_lv) & (df['rank'] <= tgt_lv)]))

    target_df = df[(df['rank'] > cur_lv) & (df['rank'] <= tgt_lv)].copy()
    beyond_df = df[df['rank'] > tgt_lv].copy()
    top_n_df = target_df.head(top_n)

    tabs = st.tabs([f"🔥 Top {len(top_n_df)}", "🟡 重点词", "🔴 超纲词", "🟢 已掌握"])

    def render_vocab_tab(tab, data_df, key_prefix):
        with tab:
            if data_df.empty:
                st.info("无单词")
                return
            
            # 这里是之前报错的关键点：移除 from app import ...
            # 并在 render_vocab_tab 内部使用本地变量
            words_list = data_df['raw'].tolist()
            with st.expander("👁️ 单词预览"):
                st.code("\n".join([f"{r['word']} [{int(r['rank'])}]" for _, r in data_df.iterrows()]))
            
            st.divider()
            exp_fmt = st.radio("导出格式", ["TXT", "CSV"], horizontal=True, key=f"f_{key_prefix}")
            
            # 使用 container 包装按钮以防布局错乱
            with st.container():
                if st.button(f"⚡ AI 生成 {len(data_df)} 个单词的卡片", key=f"b_{key_prefix}"):
                    p_bar = st.progress(0)
                    prompt = get_base_prompt_template(exp_fmt)
                    result = call_deepseek_api_chunked(prompt, words_list, p_bar, st.empty())
                    
                    if result and "❌" not in result:
                        st.download_button("📥 下载文件", result.encode('utf-8-sig'), 
                                         file_name=f"anki_{key_prefix}.{exp_fmt.lower()}", type="primary")
                        st.code(result)
                    else: st.error(result)

    render_vocab_tab(tabs[0], top_n_df, "top")
    render_vocab_tab(tabs[1], target_df, "target")
    render_vocab_tab(tabs[2], beyond_df, "beyond")
    render_vocab_tab(tabs[3], df[df['rank'] <= cur_lv], "known")