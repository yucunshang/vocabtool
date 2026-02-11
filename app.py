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
    st.error("⚠️ 缺少文件处理依赖。请在终端运行: pip install PyPDF2 python-docx")

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 16px !important; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    [data-testid="stMetricValue"] { font-size: 28px !important; color: var(--primary-color) !important; }
    .param-box { background-color: var(--secondary-background-color); padding: 15px 20px 5px 20px; border-radius: 10px; border: 1px solid var(--border-color-light); margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据与 NLP 初始化
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
    for pkg in ['averaged_perceptron_tagger', 'punkt']:
        try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except: pass
setup_nltk()

def get_lemma(w):
    lemmas_dict = lemminflect.getAllLemmas(w)
    if not lemmas_dict: return w.lower()
    for pos in ['ADJ', 'ADV', 'VERB', 'NOUN']:
        if pos in lemmas_dict: return lemmas_dict[pos][0]
    return list(lemmas_dict.values())[0][0]

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
        except: pass
    
    for word, rank in BUILTIN_PATCH_VOCAB.items(): vocab[word] = rank
    
    # 手动降级/覆盖特定单词的权重 (将常用数字词/序数词强行压到 1000 以内)
    URGENT_OVERRIDES = {
        "china": 400, "turkey": 1500, "march": 500, "may": 100, "august": 1500, "polish": 2500,
        "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, "saturday": 300, "sunday": 300,
        "january": 400, "february": 400, "april": 400, "june": 400, "july": 400, "september": 400, "october": 400, "november": 400, "december": 400,
        "usa": 200, "uk": 200, "google": 1000, "apple": 1000, "microsoft": 1500
    }
    
    # 追加：常见基数词与序数词降级名单
    number_words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
        "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth"
    ]
    for nw in number_words:
        URGENT_OVERRIDES[nw] = 1000

    for word, rank in URGENT_OVERRIDES.items(): vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 3. 文档解析 & 动态 AI 提示词引擎
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
        st.error(f"文件解析失败: {e}")
        return ""
    return ""

def get_dynamic_prompt_template(export_format, front_style, add_pos, def_lang, ex_count, add_ety, split_polysemy):
    """
    动态生成 Anki 极速制卡 Prompt (带严格语义约束模式)
    """
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
    elif ex_count == 1:
        ex_desc = "<br><br><em>Italicized example sentence</em>"
    else:
        examples = [f"{i+1}. <em>Italicized example sentence {i+1}</em>" for i in range(ex_count)]
        ex_desc = "<br><br>" + " <br><br> ".join(examples)

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

# Content Formatting
- Column 1 (Front): {front_desc} Do NOT bold or highlight the target word.
- Column 2 (Back): Must be exactly formatted as follows (using HTML tags):
  {def_desc}{ex_desc}{ety_desc}

# Action
Process the following list of words immediately and output ONLY the final code block:"""

    return prompt

# ==========================================
# 4. 多核并发 API 引擎 (核心极速区)
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
    
    try:
        for attempt in range(3):
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            if resp.status_code == 429: 
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 402: return "❌ ERROR_402_NO_BALANCE"
            elif resp.status_code == 401: return "❌ ERROR_401_INVALID_KEY"
            resp.raise_for_status()
            
            result = resp.json()['choices'][0]['message']['content'].strip()
            
            if result.startswith("```"):
                lines = result.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                result = '\n'.join(lines).strip()
            return result
            
        return f"\n🚨 批次超时或被限流，此批次 ({len(batch_words)}词) 生成失败。"
    except Exception as e:
        return f"\n🚨 批次请求发生异常: {str(e)}"

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
# 5. 分析引擎
# ==========================================
def analyze_words(unique_word_list):
    unique_items = [] 
    JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're'}
    for item_lower in unique_word_list:
        if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
        if item_lower in JUNK_WORDS: continue
        actual_rank = vocab_dict.get(item_lower, 99999)
        
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            unique_items.append({"word": f"{item_lower} ({domain})", "rank": term_rank, "raw": item_lower})
            continue
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_lower.title())
            unique_items.append({"word": display, "rank": actual_rank, "raw": item_lower})
            continue
        if actual_rank != 99999:
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 6. UI 与流水线状态管理
# ==========================================
st.title("🚀 Vocab Master Pro - Stable")
st.markdown("💡 支持粘贴长文或直接上传 `TXT / PDF / DOCX / EPUB` 文件，并**内置免费 AI** 一键生成 Anki 记忆卡片。")

if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False
    if 'base_df' in st.session_state: del st.session_state.base_df

# === 更新：按照截图设置参数默认值 ===
st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: current_level = st.number_input("🎯 当前词汇量 (起)", 0, 30000, 9000, 500)     # 修改默认值为 9000
with c2: target_level = st.number_input("🎯 目标词汇量 (止)", 0, 30000, 15000, 500)    # 修改默认值为 15000
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 100, 10)                 # 修改默认值为 100
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 10000, 500) # 修改默认值为 10000
with c5: 
    st.write("") 
    st.write("") 
    show_rank = st.checkbox("🔢 附加显示 Rank", value=True)
st.markdown("</div>", unsafe_allow_html=True)

col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本 (支持10万字以内)", height=150, key="raw_input_text")
with col_input2:
    st.info("💡 **多格式解析**：直接拖入电子书/论文原著 👇")
    uploaded_file = st.file_uploader("📂 上传文档", type=["txt", "pdf", "docx", "epub"], key=f"uploader_{st.session_state.uploader_key}")

col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1: btn_process = st.button("🚀 极速智能解析", type="primary", use_container_width=True)
with col_btn2: st.button("🗑️ 一键清空", on_click=clear_all_inputs, use_container_width=True)

st.divider()

# ==========================================
# 7. 后台硬核计算
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
            raw_words = re.findall(r"[a-zA-Z']+", combined_text)
            unique_raw_words = list(set(raw_words)) 
            lemmatized_unique = [get_lemma(w).lower() for w in unique_raw_words]
            unique_lemmas = list(set(lemmatized_unique)) 
            
            st.session_state.base_df = analyze_words(unique_lemmas)
            
            st.session_state.stats = {
                "raw_count": len(raw_words),
                "unique_count": len(unique_lemmas),
                "valid_count": len(st.session_state.base_df),
                "time": time.time() - start_time
            }
            st.session_state.is_processed = True

# ==========================================
# 8. 动态界面渲染
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
        df = df.sort_values(by='rank')
        top_df = df[df['rank'] >= min_rank_threshold].sort_values(by='rank', ascending=True).head(top_n)
        
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
                            display_lines.append(f"{row['word']} [Rank: {rank_str}]")
                        else:
                            display_lines.append(row['word'])
                    
                    with st.expander("👁️ 查看单词列表", expanded=expand_default):
                        st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制单词</p>", unsafe_allow_html=True)
                        st.code("\n".join(display_lines), language='text')
                    
                    st.divider()
                    
                    st.markdown("#### ⚙️ 定制卡片内容")
                    ui_col1, ui_col2 = st.columns(2)
                    
                    # === 更新：按照截图设置制卡界面的默认单选框和复选框 ===
                    with ui_col1:
                        st.markdown("**正面配置 (Front)**")
                        export_format = st.radio("输出格式:", ["TXT", "CSV"], horizontal=True, key=f"fmt_{df_key}", index=0)
                        ui_front = st.radio("呈现形式:", ["短语/搭配 (Phrase)", "仅单词 (Word Only)"], horizontal=True, key=f"front_{df_key}", index=0)
                        ui_pos = st.checkbox("附加词性标示 (如 v, n)", value=True, key=f"pos_{df_key}")
                        ui_poly = st.radio("多义词处理:", ["拆分为多张卡片 (原版默认)", "仅生成核心释义 (1词1卡)"], index=1, horizontal=True, key=f"poly_{df_key}") # 默认选中第二项

                    with ui_col2:
                        st.markdown("**背面配置 (Back)**")
                        ui_def = st.radio("释义语言:", ["纯英文 (EN)", "纯中文 (ZH)", "中英双语 (EN+ZH)"], index=0, horizontal=True, key=f"def_{df_key}") # 默认选中纯英文
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