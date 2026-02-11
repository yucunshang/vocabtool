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

# 尝试导入多格式文档处理库
try:
    import PyPDF2
    import docx
except ImportError:
    pass

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro V5", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 16px !important; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #007bff !important; }
    /* 参数区域样式优化 */
    .param-container { border-bottom: 1px solid #eee; padding-bottom: 20px; margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API Key 获取 (严格遵循原始设置)
# ==========================================
try:
    # 直接读取 secrets，不做任何 UI 展示
    user_api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    st.error("❌ 未检测到 API Key配置。请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
    st.stop()

# ==========================================
# 3. 数据与 NLP 初始化 (保持健壮版)
# ==========================================
@st.cache_data
def load_knowledge_base():
    data = {"terms": {}, "proper": {}, "patch": {}, "ambiguous": set()}
    try:
        if os.path.exists('data/terms.json'):
            with open('data/terms.json', 'r', encoding='utf-8') as f: data["terms"] = {k.lower(): v for k, v in json.load(f).items()}
        if os.path.exists('data/proper.json'):
            with open('data/proper.json', 'r', encoding='utf-8') as f: data["proper"] = {k.lower(): v for k, v in json.load(f).items()}
        if os.path.exists('data/patch.json'):
            with open('data/patch.json', 'r', encoding='utf-8') as f: data["patch"] = json.load(f)
        if os.path.exists('data/ambiguous.json'):
            with open('data/ambiguous.json', 'r', encoding='utf-8') as f: data["ambiguous"] = set(json.load(f))
    except Exception: pass
    return data["terms"], data["proper"], data["patch"], data["ambiguous"]

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource
def setup_nltk():
    try: nltk.data.find('corpora/wordnet')
    except LookupError:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'wordnet']:
            try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
            except: pass

setup_nltk()

def get_lemma(w):
    if not w: return ""
    try:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict: return w.lower()
        for pos in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            if pos in lemmas_dict: return lemmas_dict[pos][0]
        return list(lemmas_dict.values())[0][0]
    except: return w.lower()

@st.cache_data
def load_vocab():
    vocab = {}
    file_path = next((f for f in ["coca_cleaned.csv", "data.csv"] if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
            df = df.sort_values(r_col, ascending=True).drop_duplicates(subset=[w_col], keep='first')
            vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except Exception: pass
    
    if BUILTIN_PATCH_VOCAB:
        for word, rank in BUILTIN_PATCH_VOCAB.items(): vocab[word] = rank
    
    URGENT_OVERRIDES = {"china": 400, "usa": 200, "uk": 200, "google": 1000, "apple": 1000}
    for word, rank in URGENT_OVERRIDES.items(): vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 4. 文档解析 & 并发 API (线程安全)
# ==========================================
def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    uploaded_file.seek(0)
    try:
        if ext == 'txt': return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            doc = docx.Document(uploaded_file)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == 'epub': return "EPUB解析暂略" # 简化展示
    except Exception: return ""
    return ""

def get_base_prompt_template(export_format="TXT"):
    return f"""【角色设定】 你是一位精通词源学、认知心理学以及 Anki 算法的“英语词汇专家与闪卡制作大师”。
1. 核心原则：原子性 (Atomicity)
若一个单词有多个常用含义，必须拆分为多条独立数据。
2. 卡片正面 (Column 1)
提供自然的短语或搭配 (Phrase/Collocation)。
3. 卡片背面 (Column 2 - 整合页)
使用 HTML 标签排版，包含三个部分，用 <br><br> 分隔：
英文释义 <br><br> <em>斜体例句</em> <br><br> 【中文词源/记忆法】
4. 输出格式标准 ({export_format} 格式)
纯文本代码块，无 Markdown 包裹。逗号分隔，字段用双引号包裹。
"""

def _fetch_deepseek_chunk_safe(batch_data):
    index, batch_words, prompt_template, api_key = batch_data
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    system_enforcement = "\n\n【系统绝对强制指令】直接输出最终的数据代码，不要回复“好的”，不要使用 ```csv 包裹！"
    full_prompt = f"{prompt_template}{system_enforcement}\n\n待处理单词列表：\n{', '.join(batch_words)}"
    
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": full_prompt}], "temperature": 0.3, "max_tokens": 4096}
    
    try:
        for attempt in range(3):
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429: 
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code != 200: return (index, "", f"HTTP {resp.status_code}")
            
            result = resp.json()['choices'][0]['message']['content'].strip()
            if result.startswith("```"):
                lines = result.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                result = '\n'.join(lines).strip()
            return (index, result, None)
        return (index, "", "TIMEOUT")
    except Exception as e: return (index, "", str(e))

def run_concurrent_api(words, prompt_template, api_key, progress_bar, status_text):
    MAX_WORDS = 300 
    words = words[:MAX_WORDS]
    CHUNK_SIZE = 30
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    tasks = [(i, chunk, prompt_template, api_key) for i, chunk in enumerate(chunks)]
    results_map = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {executor.submit(_fetch_deepseek_chunk_safe, task): task[0] for task in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, res_str, err = future.result()
            if not err: results_map[idx] = res_str
            completed += 1
            progress_bar.progress(completed / len(chunks))
            status_text.markdown(f"⚡ AI 正在处理第 {completed}/{len(chunks)} 批数据...")

    final_output = []
    for i in range(len(chunks)):
        if i in results_map: final_output.append(results_map[i])
    return "\n".join(final_output)

def analyze_words(unique_word_list, min_rank):
    unique_items = [] 
    STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it'}
    
    for item_lower in unique_word_list:
        if len(item_lower) < 2 or item_lower in STOP_WORDS: continue
        
        actual_rank = vocab_dict.get(item_lower, 99999)
        # 严格执行 rank 过滤
        if actual_rank < min_rank and actual_rank != 99999: continue

        if item_lower in BUILTIN_TECHNICAL_TERMS:
             unique_items.append({"word": f"{item_lower}", "rank": actual_rank, "raw": item_lower})
        elif actual_rank != 99999:
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower})
        elif item_lower in PROPER_NOUNS_DB:
             unique_items.append({"word": item_lower, "rank": 99999, "raw": item_lower})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 5. UI 布局 (无侧边栏，参数常驻)
# ==========================================
st.title("🚀 Vocab Master Pro - V5")

# 初始化 Session State
if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False
if "generated_cards" not in st.session_state: st.session_state.generated_cards = {} 

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False
    st.session_state.generated_cards = {}

# --- 参数设置区域 (显式展示，不折叠) ---
st.markdown("### ⚙️ 核心参数")
c1, c2, c3, c4 = st.columns(4)
with c1: current_level = st.number_input("🎯 当前词汇量 (起)", 0, 30000, 4500, 500, help="低于此排名的词将被视为‘熟词’")
with c2: target_level = st.number_input("🎯 目标词汇量 (止)", 0, 30000, 15000, 500, help="高于此排名的词将被视为‘超纲’")
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 50, 10)
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 1000, 500, help="直接过滤掉排名极高(太简单)的词")
show_rank = st.checkbox("在列表中显示词频 Rank", value=True)

st.divider()

# --- 输入区 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本", height=150, key="raw_input_text", placeholder="在此粘贴英文内容...")
with col_input2:
    st.markdown("#### 📂 文档解析")
    uploaded_file = st.file_uploader("支持 TXT, PDF, DOCX", type=["txt", "pdf", "docx"], key=f"uploader_{st.session_state.uploader_key}")

col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1: btn_process = st.button("🚀 开始分析", type="primary", use_container_width=True)
with col_btn2: st.button("🗑️ 清空", on_click=clear_all_inputs, use_container_width=True)

# ==========================================
# 6. 处理与展示逻辑
# ==========================================
if btn_process:
    with st.spinner("🧠 分析中..."):
        start_time = time.time()
        combined_text = raw_text
        if uploaded_file is not None: combined_text += "\n" + extract_text_from_file(uploaded_file)
            
        if not combined_text.strip():
            st.warning("⚠️ 内容为空")
        else:
            raw_words = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", combined_text)
            lemmatized_words = [get_lemma(w) for w in raw_words]
            unique_lemmas = list(set([w.lower() for w in lemmatized_words]))
            
            # 将 min_rank_threshold 传入分析函数
            st.session_state.base_df = analyze_words(unique_lemmas, min_rank_threshold)
            st.session_state.lemma_text = " ".join(lemmatized_words)
            st.session_state.stats = {
                "raw_count": len(raw_words),
                "unique_count": len(unique_lemmas),
                "valid_count": len(st.session_state.base_df),
                "time": time.time() - start_time
            }
            st.session_state.is_processed = True
            st.session_state.generated_cards = {} 

if st.session_state.get("is_processed", False):
    stats = st.session_state.stats
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总词数", f"{stats['raw_count']:,}")
        c2.metric("去重后", f"{stats['unique_count']:,}")
        c3.metric("有效词", f"{stats['valid_count']:,}")
        c4.metric("耗时", f"{stats['time']:.2f}s")
    
    df = st.session_state.base_df.copy()
    
    if not df.empty:
        def categorize(row):
            r = row['rank']
            if r <= current_level: return "known"
            elif r <= target_level: return "target"
            else: return "beyond"
        
        df['final_cat'] = df.apply(categorize, axis=1)
        df = df.sort_values(by='rank')
        
        top_df = df[df['rank'] >= min_rank_threshold].sort_values(by='rank', ascending=True).head(top_n)
        target_df = df[df['final_cat']=='target']
        beyond_df = df[df['final_cat']=='beyond']
        
        tabs = st.tabs(["🔥 Top精选", "🟡 重点词汇", "🔴 超纲词汇", "📝 原文下载"])
        
        def render_word_tab(tab_obj, data_df, tab_key):
            with tab_obj:
                if data_df.empty:
                    st.info("该区间暂无单词")
                    return

                col_list, col_ai = st.columns([1, 2])
                with col_list:
                    st.markdown(f"**单词预览 ({len(data_df)})**")
                    display_text = []
                    for _, row in data_df.iterrows():
                        suffix = f" [{int(row['rank'])}]" if show_rank and row['rank'] != 99999 else ""
                        display_text.append(f"{row['word']}{suffix}")
                    st.text_area("列表", value="\n".join(display_text), height=400, label_visibility="collapsed")

                with col_ai:
                    st.markdown("#### 🤖 AI 卡片制作")
                    export_fmt = st.radio("格式", ["TXT", "CSV"], horizontal=True, key=f"fmt_{tab_key}")
                    pure_words = data_df['word'].tolist()
                    
                    # 恢复：API直接调用和手动复制Prompt的双Tab设计
                    ai_tab1, ai_tab2 = st.tabs(["⚡ 一键调用 DeepSeek", "📋 手动复制 Prompt"])
                    
                    with ai_tab1:
                        res_key = f"{tab_key}_{export_fmt}"
                        if st.session_state.generated_cards.get(res_key):
                            st.success("✅ 已生成")
                            st.download_button("📥 下载结果", st.session_state.generated_cards[res_key], f"anki_{tab_key}.{export_fmt.lower()}")
                            st.code(st.session_state.generated_cards[res_key], language="text")
                        else:
                            if st.button(f"⚡ 生成 {tab_key}", key=f"btn_{tab_key}"):
                                p_bar = st.progress(0)
                                s_text = st.empty()
                                res = run_concurrent_api(pure_words, get_base_prompt_template(export_fmt), user_api_key, p_bar, s_text)
                                st.session_state.generated_cards[res_key] = res
                                st.rerun()

                    with ai_tab2:
                        st.info("💡 如果您想使用 ChatGPT/Claude 等自己的 AI 工具，请点击右上角一键复制下方完整指令：")
                        full_prompt_to_copy = f"{get_base_prompt_template(export_fmt)}\n\n待处理单词：\n{', '.join(pure_words)}"
                        st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制</p>", unsafe_allow_html=True)
                        st.code(full_prompt_to_copy, language='markdown')

        render_word_tab(tabs[0], top_df, "top")
        render_word_tab(tabs[1], target_df, "target")
        render_word_tab(tabs[2], beyond_df, "beyond")
        
        with tabs[3]:
            st.info("💡 这是自动词形还原后的全文输出，已针对长文优化防卡死体验。")
            st.download_button("💾 下载原文", st.session_state.lemma_text, "lemmatized.txt")
            st.text_area("预览", st.session_state.lemma_text[:2000], height=300)