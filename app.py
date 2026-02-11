import streamlit as st
import pandas as pd
import re
import os
import json
import time
import requests
import zipfile
import concurrent.futures
import lemminflect
import nltk
from collections import Counter  # <--- [新增] 引入计数器

# ==========================================
# 0. 尝试导入多格式文档处理库
# ==========================================
try:
    import PyPDF2
    import docx
except ImportError:
    st.error("⚠️ 缺少文件处理依赖。请在终端运行: pip install PyPDF2 python-docx")

# ==========================================
# 1. 基础 UI 配置与 State 初始化
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 16px !important; }
    header {visibility: hidden;} footer {visibility: hidden;}\
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    [data-testid="stMetricValue"] { font-size: 28px !important; color: var(--primary-color) !important; }
    .param-box { background-color: var(--secondary-background-color); padding: 15px 20px 5px 20px; border-radius: 10px; border: 1px solid var(--border-color-light); margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
</style>
""", unsafe_allow_html=True)

# 统一初始化 Session State (提升健壮性)
if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False
if "base_df" not in st.session_state: st.session_state.base_df = pd.DataFrame()
if "stats" not in st.session_state: st.session_state.stats = {}

# ==========================================
# 2. 全局核心配置字典 (集中管理，提升拓展性)
# ==========================================
# 既是人名又是核心单词的“免死金牌”白名单
SAFE_NAMES_DB = {
    'will', 'mark', 'rose', 'lily', 'bill', 'pat', 'joy', 'hope', 'penny', 'faith', 
    'grace', 'amber', 'crystal', 'dawn', 'eve', 'holly', 'ivy', 'robin', 'summer', 
    'autumn', 'winter', 'brook', 'stone', 'cliff', 'ash', 'art', 'frank', 'grant', 
    'miles', 'ward', 'dean', 'earl', 'duke', 'king', 'prince', 'baker', 'smith', 
    'foster', 'clark', 'cook', 'bell', 'hill', 'wood', 'ray', 'guy', 'max', 
    'page', 'rusty', 'cash', 'chance', 'clay', 'fox', 'lane', 'reed', 'roman', 'tanner', 
    'paris', 'london', 'chase', 'hunter', 'drake', 'drew', 'buck', 'buddy', 'chuck', 
    'colt', 'daisy', 'dash', 'destiny', 'diamond', 'dusty', 'echo', 'ember', 'fern', 
    'flint', 'flora', 'gale', 'gene', 'harmony', 'hazel', 'heather', 'iris', 'jade', 
    'jasmine', 'jewel', 'justice', 'laurel', 'marina', 'melody', 'olive', 'opal', 
    'pierce', 'piper', 'poppy', 'rex', 'ruby', 'sage', 'savannah', 'scarlett', 'scout', 
    'sienna', 'sierra', 'skip', 'sky', 'starr', 'trinity', 'victor', 'violet', 'wade', 
    'willow', 'woody', 'wren', 'brown', 'white', 'black', 'green', 'young', 'hall', 
    'wright', 'scott', 'price', 'long', 'major', 'rich', 'dick', 'christian', 'kelly', 'parker'
}

# 强行覆盖的词汇等级矩阵 (地名/节日/月份/大厂/数字)
GLOBAL_ENTITY_RANKS = {
    "africa": 1000, "asia": 1000, "europe": 800, "america": 500, "australia": 1500, "antarctica": 4000,
    "china": 400, "usa": 200, "uk": 200, "britain": 800, "england": 800, "france": 800, "germany": 900, "japan": 900, "russia": 900, "india": 1000, "italy": 1000, "canada": 1000, "spain": 1200, "mexico": 1200, "brazil": 1500, "korea": 1500, "egypt": 2000, "greece": 2000, "ireland": 2000, "scotland": 2000, "wales": 2500, "sweden": 2500, "switzerland": 2500, "norway": 3000, "denmark": 3000, "finland": 3000, "poland": 2500, "netherlands": 2500, "portugal": 3000, "vietnam": 3000, "thailand": 3000, "singapore": 3000, "malaysia": 3000, "indonesia": 3000, "philippines": 3000, "turkey": 1500, "israel": 1500, "iran": 2000, "iraq": 2000,
    "american": 300, "british": 500, "english": 300, "french": 600, "german": 700, "chinese": 800, "japanese": 800, "russian": 900, "indian": 900, "italian": 1000, "spanish": 1000, "canadian": 1200, "korean": 1500, "arabic": 2000, "latin": 2000, "greek": 2000,
    "london": 800, "paris": 1000, "tokyo": 1500, "rome": 1500, "berlin": 2000, "moscow": 2000, "beijing": 2500, "shanghai": 2500, "washington": 500, "york": 500, "chicago": 1500, "boston": 1500, "sydney": 2000,
    "christmas": 800, "easter": 2000, "halloween": 2500, "thanksgiving": 1500, "valentine": 3000, "hanukkah": 5000, "ramadan": 5000, "diwali": 6000, "carnival": 4000, "festival": 1500, "holiday": 1000,
    "jewish": 1500, "muslim": 1500, "christian": 1500, "catholic": 1500, "protestant": 2500, "hindu": 3000, "buddhist": 3000, "islam": 2000, "buddhism": 3500, "christianity": 2000,
    "google": 1000, "apple": 1000, "microsoft": 1500, "facebook": 1500, "twitter": 2000, "amazon": 1500,
    "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, "saturday": 300, "sunday": 300,
    "january": 400, "february": 400, "march": 400, "april": 400, "may": 100, "june": 400, "july": 400, "august": 1500, "september": 400, "october": 400, "november": 400, "december": 400
}

# 基础数字词写入全局矩阵
for _nw in ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth"]:
    GLOBAL_ENTITY_RANKS[_nw] = 1000

# ==========================================
# 3. 数据与 NLP 初始化 (带容错机制)
# ==========================================
@st.cache_data
def load_knowledge_base():
    try:
        if not os.path.exists('data'):
            return {}, {}, {}, set()
        with open('data/terms.json', 'r', encoding='utf-8') as f: terms = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/proper.json', 'r', encoding='utf-8') as f: proper = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/patch.json', 'r', encoding='utf-8') as f: patch = json.load(f)
        with open('data/ambiguous.json', 'r', encoding='utf-8') as f: ambiguous = set(json.load(f))
        return terms, proper, patch, ambiguous
    except Exception as e:
        print(f"Knowledge base load error: {e}")
        return {}, {}, {}, set()

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource
def setup_nltk():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    # 防御性下载，即使失败也不抛出异常
    for pkg in ['averaged_perceptron_tagger', 'punkt', 'names']:
        try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except Exception: pass
setup_nltk()

@st.cache_data
def load_names_db():
    try:
        from nltk.corpus import names
        return set([n.lower() for n in names.words()])
    except Exception:
        # 如果 nltk 缺失或加载失败，返回空集以保证主体程序继续运行
        return set()
NLTK_NAMES_DB = load_names_db()

def get_lemma(w):
    try:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict: return w.lower()
        for pos in ['ADJ', 'ADV', 'VERB', 'NOUN']:
            if pos in lemmas_dict: return lemmas_dict[pos][0]
        return list(lemmas_dict.values())[0][0]
    except:
        return w.lower()

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
        except Exception as e: 
            print(f"Vocab CSV load error: {e}")
    
    # 按照优先级合并词库: 基础 CSV < 补丁数据 < 强制常量映射
    for word, rank in BUILTIN_PATCH_VOCAB.items(): vocab[word] = rank
    for word, rank in GLOBAL_ENTITY_RANKS.items(): vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 4. 文档解析 & AI 提示词引擎
# ==========================================
def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    uploaded_file.seek(0)
    try:
        if ext == 'txt':
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            doc = docx.Document(uploaded_file)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == 'epub':
            text_blocks = []
            with zipfile.ZipFile(uploaded_file) as z:
                for filename in z.namelist():
                    if filename.endswith(('.html', '.xhtml', '.htm', '.xml')):
                        try:
                            content = z.read(filename).decode('utf-8', errors='ignore')
                            clean_text = re.sub(r'<[^>]+>', ' ', content)
                            text_blocks.append(clean_text)
                        except: pass
            return " ".join(text_blocks)
    except Exception as e:
        st.error(f"⚠️ 文件解析失败: {e}")
        return ""
    return ""

def get_dynamic_prompt_template(export_format, front_style, add_pos, def_lang, ex_count, add_ety, split_polysemy):
    front_desc = "A natural phrase or collocation using the specific meaning." if front_style == "phrase" else "The target word itself."
    if add_pos:
        front_desc += " MUST append the precise part of speech tag at the end, e.g., ' (v)', ' (n)', ' (adj)'."
    else:
        front_desc += " Do NOT add part of speech tags."

    def_map = {
        "en": "English definition of the specific meaning",
        "zh": "Chinese definition of the specific meaning",
        "en_zh": "English definition followed by Chinese definition separated by a slash (/)"
    }
    def_desc = def_map.get(def_lang, "English definition")

    if ex_count == 0:
        ex_desc = ""
        ex_rule = "Generate ZERO example sentences. Do NOT include any examples."
    elif ex_count == 1:
        ex_desc = "<br><br><em>Italicized example sentence</em>"
        ex_rule = "Generate EXACTLY ONE example sentence. NEVER generate two or more examples."
    else:
        examples = [f"{i+1}. <em>Italicized example sentence {i+1}</em>" for i in range(ex_count)]
        ex_desc = "<br><br>" + " <br><br> ".join(examples)
        ex_rule = f"Generate EXACTLY {ex_count} example sentences, numbered as shown."

    ety_desc = "<br><br>【词根词缀/词源】Chinese etymology or affix explanation." if add_ety else ""
    
    if split_polysemy:
        poly_rule = "Atomicity: ONE meaning per row. Polysemous words MUST be split into multiple separate rows. NEVER stack multiple definitions in one card."
    else:
        poly_rule = "One Card Per Word: Generate EXACTLY ONE row per input word. Extract ONLY the single most common/primary meaning. NEVER split a word into multiple cards."

    prompt = f"""# Role
You are an expert English linguist and a highly precise Anki flashcard generator.

# Task
Process the user's input words, auto-correct any spelling errors/abbreviations, and generate Anki flashcards strictly following the rules below.

# Strict Rules
1. Format: Pure {export_format} format in a single code block. NO conversational filler, NO markdown formatting outside the code block.
2. Structure: STRICTLY TWO COLUMNS per row. Format: "Column 1","Column 2"
3. Quotes: Both columns MUST be wrapped in double quotes. Use single quotes (' ') inside the text if needed.
4. {poly_rule}
5. Strict Alignment (CRITICAL): The generated phrase, part of speech (if requested), definition, example sentence(s), and etymology MUST strictly logically align with the EXACT SAME specific meaning of the target word. Do not mix definitions or examples of different meanings in a single card.
6. Example Count Constraint (CRITICAL): {ex_rule}

# Content Formatting
- Column 1 (Front): {front_desc} Do NOT bold or highlight the target word.
- Column 2 (Back): Must be exactly formatted as follows (using HTML tags):
  {def_desc}{ex_desc}{ety_desc}

# Action
Process the following list of words immediately and output ONLY the final code block:"""

    return prompt

# ==========================================
# 5. 多核并发 API 引擎 (健壮性升级版)
# ==========================================
def _fetch_deepseek_chunk(batch_words, prompt_template, api_key):
    url = "https://api.deepseek.com/chat/completions".strip()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    system_enforcement = "\n\n【系统绝对强制指令】现在我已经发送了单词列表，请立即且直接输出最终的数据代码，绝对不准回复“好的”、“没问题”等任何客套话，绝对不准使用 ```csv 等 Markdown 语法包裹代码！"
    full_prompt = f"{prompt_template}{system_enforcement}\n\n待处理单词列表：\n{', '.join(batch_words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3,
        "max_tokens": 4096
    }
    
    for attempt in range(4): # 增加重试次数增强网络稳定性
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 429: 
                time.sleep(3 * (attempt + 1)) # 指数退避，防止被封IP
                continue
            if resp.status_code == 402: return "❌ ERROR_402_NO_BALANCE"
            elif resp.status_code == 401: return "❌ ERROR_401_INVALID_KEY"
            resp.raise_for_status()
            
            result = resp.json()['choices'][0]['message']['content'].strip()
            
            # 使用 Regex 正则强力清洗 Markdown 标签 (极高稳定性保障)
            result = re.sub(r"^```(?:csv|txt|text)?\n", "", result, flags=re.IGNORECASE)
            result = re.sub(r"\n```$", "", result)
            return result.strip()
            
        except Exception as e:
            if attempt == 3:
                return f"\n🚨 批次请求异常: {str(e)}"
            time.sleep(2)
            
    return f"\n🚨 批次被限流，此批次 ({len(batch_words)}词) 生成失败。"

def call_deepseek_api_chunked(prompt_template, words, progress_bar, status_text):
    try: api_key = st.secrets["DEEPSEEK_API_KEY"]
    except KeyError: return "⚠️ 站长配置错误：未在 Streamlit 后台 Secrets 中配置 DEEPSEEK_API_KEY。"
    
    if not words: return "⚠️ 错误：没有需要生成的单词。"
    
    MAX_WORDS = 250
    if len(words) > MAX_WORDS:
        st.warning(f"⚠️ 为保证并发稳定，本次仅截取前 **{MAX_WORDS}** 个单词。")
        words = words[:MAX_WORDS]

    CHUNK_SIZE = 30  
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    total_words = len(words)
    processed_count = 0
    results_ordered = [None] * len(chunks)
    
    status_text.markdown("🚀 **并发任务已发射！** 正在全速生成首批卡片（首次返回约需 8~12 秒，请稍候）...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(_fetch_deepseek_chunk, chunk, prompt_template, api_key): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            chunk_size = len(chunks[idx])
            res = future.result()
            
            if "ERROR_402_NO_BALANCE" in res: return "❌ 错误：DeepSeek 账户余额不足，请充值。"
            if "ERROR_401_INVALID_KEY" in res: return "❌ 错误：API Key 无效。"
            
            results_ordered[idx] = res 
            
            processed_count += chunk_size
            current_progress = min(processed_count / total_words, 1.0)
            progress_bar.progress(current_progress)
            status_text.markdown(f"**⚡ AI 多核并发全速编纂中：** `{processed_count} / {total_words}` 词")

    return "\n".join(filter(None, results_ordered))

# ==========================================
# 6. 分析引擎 (内置无感知人名过滤拦截器) - [已修改支持词频统计]
# ==========================================
def analyze_words(unique_word_list, freq_dict): # <--- [修改] 增加 freq_dict 参数
    unique_items = [] 
    JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're', 'don', 'doesn', 'didn', 'won', 'isn', 'aren', 'ain'}
    
    for item_lower in unique_word_list:
        if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
        if item_lower in JUNK_WORDS: continue
        
        # 获取该词在本文中的频率
        doc_freq = freq_dict.get(item_lower, 1) # <--- [新增] 获取词频

        # 🛡️ 核心隐形拦截：强制人名过滤
        if item_lower in NLTK_NAMES_DB and item_lower not in SAFE_NAMES_DB:
            continue

        actual_rank = vocab_dict.get(item_lower, 99999)
        
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            # [修改] 增加 freq 字段
            unique_items.append({"word": f"{item_lower} ({domain})", "rank": term_rank, "raw": item_lower, "freq": doc_freq})
            continue
            
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_lower.title())
            # [修改] 增加 freq 字段
            unique_items.append({"word": display, "rank": actual_rank, "raw": item_lower, "freq": doc_freq})
            continue
            
        if actual_rank != 99999:
            # [修改] 增加 freq 字段
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower, "freq": doc_freq})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 7. UI 视图层
# ==========================================
st.title("🚀 Vocab Master Pro - Stable Release")
st.markdown("💡 支持粘贴长文或直接上传 `TXT / PDF / DOCX / EPUB` 文件，并**内置免费 AI** 一键生成 Anki 记忆卡片。 *(词频分级数据基于 COCA 20000 权威核心词库)*")

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False
    st.session_state.base_df = pd.DataFrame()

st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: current_level = st.number_input("🎯 当前词汇量 (起)", 0, 20000, 9000, 500)     
with c2: target_level = st.number_input("🎯 目标词汇量 (止)", 0, 20000, 15000, 500)    
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 100, 10)                 
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 6000, 500) 
with c5: 
    # [修改] 增加了排序逻辑的选择
    sort_mode = st.radio("📊 排序优先", ["COCA 词频 (默认)", "本文出现频率"], index=0)
    show_rank = st.checkbox("🔢 显示详细数据", value=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- UI 调整：文本框与上传文件并排 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本 (支持10万字以内)", height=150, key="raw_input_text")
with col_input2:
    uploaded_file = st.file_uploader("📂 上传文档", type=["txt", "pdf", "docx", "epub"], key=f"uploader_{st.session_state.uploader_key}")

st.button("🗑️ 一键清空", on_click=clear_all_inputs, use_container_width=True)
btn_process = st.button("🚀 极速智能解析", type="primary", use_container_width=True)

st.divider()

# ==========================================
# 8. 流水线执行 - [已修改：先还原词形再统计频率]
# ==========================================
if btn_process:
    with st.spinner("🧠 正在急速读取文件并进行智能解析（性能优化版）..."):
        start_time = time.time()
        combined_text = raw_text
        if uploaded_file is not None: combined_text += "\n" + extract_text_from_file(uploaded_file)
            
        if not combined_text.strip():
            st.warning("⚠️ 未提取到任何有效文本！")
            st.session_state.is_processed = False
        elif vocab_dict:
            # 1. 提取所有原始单词
            raw_words = re.findall(r"[a-zA-Z']+", combined_text)
            
            # 2. 全量词形还原 (为了统计 accurately，必须先还原再 count)
            # 注意：这里我们不对 raw_words 去重，而是对所有词进行还原
            all_lemmas_with_dups = [get_lemma(w).lower() for w in raw_words]
            
            # 3. 统计本文词频
            lemma_counts = Counter(all_lemmas_with_dups)
            unique_lemmas = list(lemma_counts.keys())
            
            # 4. 核心业务调用 (传入 freq_dict 即 lemma_counts)
            st.session_state.base_df = analyze_words(unique_lemmas, lemma_counts)
            
            st.session_state.stats = {
                "raw_count": len(raw_words),
                "unique_count": len(unique_lemmas),
                "valid_count": len(st.session_state.base_df),
                "time": time.time() - start_time
            }
            st.session_state.is_processed = True

# ==========================================
# 9. 动态结果渲染
# ==========================================
if st.session_state.get("is_processed", False):
    
    stats = st.session_state.stats
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric(label="📝 解析总字数", value=f"{stats['raw_count']:,}")
    col_m2.metric(label="✂️ 去重词根数", value=f"{stats['unique_count']:,}")
    col_m3.metric(label="🎯 纳入分级词汇", value=f"{stats['valid_count']:,}")
    col_m4.metric(label="⚡ 极速解析耗时", value=f"{stats['time']:.2f} 秒")
    
    df = st.session_state.base_df.copy()
    
    if not df.empty:
        def categorize(row):
            r = row['rank']
            if r <= current_level: return "known"
            elif r <= target_level: return "target"
            else: return "beyond"
        
        df['final_cat'] = df.apply(categorize, axis=1)
        
        # --- [修改] 新增排序逻辑 ---
        if "本文出现频率" in sort_mode:
            # 按频率倒序 (出现次数越多越靠前)，次要关键词按 Rank
            df = df.sort_values(by=['freq', 'rank'], ascending=[False, True])
        else:
            # 按 COCA 排名正序 (默认)
            df = df.sort_values(by='rank', ascending=True)
        # -------------------
        
        top_df = df[(df['rank'] >= min_rank_threshold) & (df['rank'] < 99999)].head(top_n)
        
        t_top, t_target, t_beyond, t_known = st.tabs([
            f"🔥 Top {len(top_df)}", 
            f"🟡 重点 ({len(df[df['final_cat']=='target'])})", 
            f"🔴 超纲 ({len(df[df['final_cat']=='beyond'])})", 
            f"🟢 已掌握 ({len(df[df['final_cat']=='known'])})"
        ])
        
        def render_tab(tab_obj, data_df, label, expand_default=False, df_key=""):
            with tab_obj:
                if not data_df.empty:
                    display_lines = []
                    for _, row in data_df.iterrows():
                        if show_rank:
                            rank_str = str(int(row['rank'])) if row['rank'] != 99999 else "未收录"
                            # [修改] 展示增加了 Freq (词频)
                            freq_str = f" | Freq: {row['freq']}"
                            display_lines.append(f"{row['word']} [Rank: {rank_str}{freq_str}]")
                        else:
                            display_lines.append(row['word'])
                    
                    with st.expander("👁️ 查看单词列表", expanded=expand_default):
                        st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制单词</p>", unsafe_allow_html=True)
                        st.code("\n".join(display_lines), language='text')
                    
                    st.divider()
                    
                    st.markdown("#### ⚙️ 定制卡片内容")
                    ui_col1, ui_col2 = st.columns(2)
                    
                    with ui_col1:
                        st.markdown("**正面配置 (Front)**")
                        export_format = st.radio("输出格式:", ["TXT", "CSV"], horizontal=True, key=f"fmt_{df_key}", index=0)
                        ui_front = st.radio("呈现形式:", ["短语/搭配 (Phrase)", "仅单词 (Word Only)"], horizontal=True, key=f"front_{df_key}", index=0)
                        ui_pos = st.checkbox("附加词性标示 (如 v, n)", value=True, key=f"pos_{df_key}")
                        ui_poly = st.radio("多义词处理:", ["拆分为多张卡片 (原版默认)", "仅生成核心释义 (1词1卡)"], index=1, horizontal=True, key=f"poly_{df_key}")

                    with ui_col2:
                        st.markdown("**背面配置 (Back)**")
                        ui_def = st.radio("释义语言:", ["纯英文 (EN)", "纯中文 (ZH)", "中英双语 (EN+ZH)"], index=0, horizontal=True, key=f"def_{df_key}")
                        ui_ex = st.slider("例句数量:", 0, 5, 1, key=f"ex_{df_key}")
                        ui_ety = st.checkbox("包含【词根词缀/词源】", value=True, key=f"ety_{df_key}")

                    front_style_val = "phrase" if "短语" in ui_front else "word"
                    def_lang_val = "en" if "纯英文" in ui_def else "zh" if "纯中文" in ui_def else "en_zh"
                    split_poly_val = True if "拆分" in ui_poly else False
                    
                    custom_prompt_text = get_dynamic_prompt_template(
                        export_format=export_format,
                        front_style=front_style_val,
                        add_pos=ui_pos,
                        def_lang=def_lang_val,
                        ex_count=ui_ex,
                        add_ety=ui_ety,
                        split_polysemy=split_poly_val
                    )
                    
                    words_to_process = data_df['raw'].tolist()

                    ai_tab1, ai_tab2 = st.tabs(["🤖 模式 1：内置 AI 并发极速直出", "📋 模式 2：复制 Prompt 给第三方 AI"])
                    
                    with ai_tab1:
                        st.info("💡 站长已为您内置专属 AI 算力。采用 **多核并发技术**，极速响应，告别卡死！")
                        
                        custom_prompt = st.text_area(
                            "📝 最终 AI Prompt (系统已根据您的设置动态生成，支持手动微调)", 
                            value=custom_prompt_text, 
                            height=380, 
                            key=f"prompt_{df_key}_{export_format}"
                        )
                        
                        st.caption("⚠️ **免责声明**：AI 生成的内容（释义、例句等）可能存在偶发的不准确或幻觉，请结合实际语境使用，建议导入前稍作复核。")
                        
                        if st.button("⚡ 召唤 DeepSeek 极速生成卡片", key=f"btn_{df_key}", type="primary"):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            status_text.markdown("**⚡ 正在连接 DeepSeek 云端算力集群...**") 
                            
                            ai_start_time = time.time()
                            ai_result = call_deepseek_api_chunked(custom_prompt, words_to_process, progress_bar, status_text)
                            ai_duration = time.time() - ai_start_time
                            
                            if "❌" in ai_result and len(ai_result) < 100:
                                st.error(ai_result)
                            else:
                                status_text.markdown(f"### 🎉 编纂全部完成！(总耗时: **{ai_duration:.2f}** 秒)")
                                
                                mime_type = "text/csv" if export_format == "CSV" else "text/plain"
                                st.download_button(
                                    label=f"📥 一键下载标准 Anki 导入文件 (.{export_format.lower()})", 
                                    data=ai_result.encode('utf-8-sig'), 
                                    file_name=f"anki_cards_{label}.{export_format.lower()}", 
                                    mime=mime_type,
                                    type="primary",
                                    use_container_width=True
                                )
                                
                                st.markdown("##### 📝 预览框")
                                st.code(ai_result, language="text")
                    
                    with ai_tab2:
                        st.info("💡 如果您想使用 ChatGPT/Claude 等自己的 AI 工具，请点击右上角一键复制下方完整指令：")
                        full_prompt_to_copy = f"{custom_prompt_text}\n\n待处理单词：\n{', '.join(words_to_process)}"
                        st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制</p>", unsafe_allow_html=True)
                        st.code(full_prompt_to_copy, language='markdown')
                else: st.info("该区间暂无单词")

        render_tab(t_top, top_df, "Top精选", expand_default=True, df_key="top") 
        render_tab(t_target, df[df['final_cat']=='target'], "重点", expand_default=False, df_key="target")
        render_tab(t_beyond, df[df['final_cat']=='beyond'], "超纲", expand_default=False, df_key="beyond")
        render_tab(t_known, df[df['final_cat']=='known'], "熟词", expand_default=False, df_key="known")