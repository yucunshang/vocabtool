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
import concurrent.futures  # 多核并发引擎

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
        # 确保 data 目录存在，如果不存在则提示
        if not os.path.exists('data'):
            # 这里可以做容错，如果没有文件返回空字典，防止报错崩溃
            return {}, {}, {}, set()
            
        with open('data/terms.json', 'r', encoding='utf-8') as f: terms = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/proper.json', 'r', encoding='utf-8') as f: proper = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/patch.json', 'r', encoding='utf-8') as f: patch = json.load(f)
        with open('data/ambiguous.json', 'r', encoding='utf-8') as f: ambiguous = set(json.load(f))
        return terms, proper, patch, ambiguous
    except Exception as e:
        # 生产环境静默失败或仅打印日志，避免弹窗吓到用户
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
    # 常用词强制覆盖 rank
    URGENT_OVERRIDES = {
        "china": 400, "turkey": 1500, "march": 500, "may": 100, "august": 1500, "polish": 2500,
        "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, "saturday": 300, "sunday": 300,
        "january": 400, "february": 400, "april": 400, "june": 400, "july": 400, "september": 400, "october": 400, "november": 400, "december": 400,
        "usa": 200, "uk": 200, "google": 1000, "apple": 1000, "microsoft": 1500
    }
    for word, rank in URGENT_OVERRIDES.items(): vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 3. 文档解析 & AI 提示词引擎
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

def get_base_prompt_template(export_format="TXT"):
    return f"""【角色设定】
你是一位精通词源学、认知心理学与 Anki 算法的英语词汇专家与制卡大师。接下来我将给你提供一个巨大的单词列表（可能多达 200+ 词）。请你开启“极限高压处理模式”，严格按照以下标准进行批量制卡。

【防偷懒与极限输出协议】（最高优先级）
1. 绝对全量输出：严禁省略、严禁跳过任何单词、严禁使用“...”或“etc.”等缩写。无论列表多长，必须逐一处理。
2. 进度锚点追踪：为了防止你失去焦点，请在每张卡片背面的最后，隐蔽地加上 HTML 注释追踪进度，格式为：。
3. 截断与无缝续写：如果你在处理过程中达到了系统的单次最大输出字数限制，请立刻停止。当我发送“继续”时，你必须且只能从上一次截断的那个字符开始继续输出，不要输出任何寒暄或抱歉的话语。
请严格按以下 5 项标准处理我提供的单词，生成 Anki 导入文件：
1. 核心原则：原子性 (Atomicity)
含义拆分：若一个单词有多个常用含义（名词 vs 动词，字面义 vs 引申义等），必须拆分为多条独立数据。
严禁堆砌：每张卡片只承载一个特定语境下的含义，不准将多个释义挤在一起。
2. 卡片正面 (Column 1: Front)
内容：提供自然的短语或搭配 (Phrase/Collocation)，而非单个孤立单词。
样式：使用纯文本，不需要加粗目标单词。
3. 卡片背面 (Column 2: Back - 整合页)
背面信息必须全部合并在第二列，并使用 HTML 标签排版，包含以下三个部分：

英文释义：简练准确。
例句：使用 <em> 标签包裹，使例句呈现斜体。
【词根词缀】：用中文进行词源、前缀、词根或后缀的拆解与记忆辅助。
换行要求：三部分之间使用 <br><br> 分隔，确保界面清晰。
结构示例：英文释义<br><br><em>斜体例句</em><br><br>【词根、词源、词缀】的中文解析
4. 输出格式标准 ({export_format} 格式)
文件规范：纯文本代码块。
分隔符：使用逗号 (Comma) 分隔字段。
引号包裹：每个字段必须用双引号 ("...") 包裹，以防内容内部的标点导致导入错误。
5. 数据清洗与优化
拼写修正：自动修正用户列表中的明显拼写错误。
缩写展开：对缩写（如 WFH, aka）在背面提供全称及解释。
💡 最终输出示例（{export_format} 内容）：
"run a business","to manage or operate a company<br><br><em>He quit his job to run a business selling handmade crafts.</em><br><br>【词源】源自古英语 rinnan（跑/流动），引申为“使机器运转”或“使业务流转”"
"go for a run","an act of running for exercise<br><br><em>I go for a run every morning before work.</em><br><br>【词源】源自古英语 rinnan（跑/流动），此处为名词用法，指“奔跑”这一动作"
导入提醒： 在 Anki 导入文件时，请务必勾选 "Allow HTML in fields" (允许在字段中使用 HTML)。"""

# ==========================================
# 4. 多核并发 API 引擎 (核心极速区)
# ==========================================
def _fetch_deepseek_chunk(batch_words, prompt_template, api_key):
    """内部工作线程：负责单一批次的极速请求"""
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
    """多线程并发控制器"""
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
    # 清除旧的分析结果
    if 'base_df' in st.session_state: del st.session_state.base_df

# --- 参数配置区 ---
st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: current_level = st.number_input("🎯 当前词汇量 (起)", 0, 30000, 7500, 500)
with c2: target_level = st.number_input("🎯 目标词汇量 (止)", 0, 30000, 15000, 500)
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 50, 10)
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 3500, 500)
with c5: 
    st.write("") 
    st.write("") 
    show_rank = st.checkbox("🔢 附加显示 Rank", value=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- 双通道多格式输入 ---
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
            # 1. 提取单词
            raw_words = re.findall(r"[a-zA-Z']+", combined_text)
            
            # 2. 词形还原 (优化：仅提取不拼接全文，大幅节省内存)
            # 使用 set 先去重再还原效率不一定高，因为 context 丢失，但这里 get_lemma 是单词处理，
            # 我们可以先对 raw_words 做 set 减少 get_lemma 调用次数 (如果单词量极大)
            # 不过为了保持频率统计的潜在准确性(虽然这里没用到频次)，直接处理列表也行。
            # 既然是 stable 优化，我们只做去重后的 lemma
            
            unique_raw_words = list(set(raw_words)) # 先去重，减少 get_lemma 调用
            lemmatized_unique = [get_lemma(w).lower() for w in unique_raw_words]
            unique_lemmas = list(set(lemmatized_unique)) # 再次去重 (run -> run, running -> run)
            
            # 3. 核心分析
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
        
        # 移除 "原文防卡死下载" Tab
        t_top, t_target, t_beyond, t_known = st.tabs([
            f"🔥 Top {len(top_df)}", 
            f"🟡 重点 ({len(df[df['final_cat']=='target'])})", 
            f"🔴 超纲 ({len(df[df['final_cat']=='beyond'])})", 
            f"🟢 已掌握 ({len(df[df['final_cat']=='known'])})"
        ])
        
        def render_tab(tab_obj, data_df, label, expand_default=False, df_key=""):
            with tab_obj:
                if not data_df.empty:
                    pure_words = data_df['word'].tolist()
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
                    
                    export_format = st.radio("⚙️ 选择输出格式:", ["TXT", "CSV"], horizontal=True, key=f"fmt_{df_key}")
                    
                    ai_tab1, ai_tab2 = st.tabs(["🤖 模式 1：内置 AI 并发极速直出", "📋 模式 2：复制 Prompt 给第三方 AI"])
                    
                    with ai_tab1:
                        st.info("💡 站长已为您内置专属 AI 算力。采用 **多核并发技术**，极速响应，告别卡死！")
                        
                        custom_prompt = st.text_area(
                            "📝 自定义 AI Prompt (可修改)", 
                            value=get_base_prompt_template(export_format), 
                            height=500, 
                            key=f"prompt_{df_key}_{export_format}"
                        )
                        
                        if st.button("⚡ 召唤 DeepSeek 极速生成卡片", key=f"btn_{df_key}", type="primary"):
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            status_text.markdown("**⚡ 正在连接 DeepSeek 云端算力集群...**") 
                            
                            # ⏳ 开始精准计时
                            ai_start_time = time.time()
                            
                            ai_result = call_deepseek_api_chunked(custom_prompt, pure_words, progress_bar, status_text)
                            
                            # ⏳ 结束精准计时
                            ai_duration = time.time() - ai_start_time
                            
                            if "❌" in ai_result and len(ai_result) < 100:
                                st.error(ai_result)
                            else:
                                # 🏅 终极跑分墙展示
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
                        full_prompt_to_copy = f"{get_base_prompt_template(export_format)}\n\n待处理单词：\n{', '.join(pure_words)}"
                        st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制</p>", unsafe_allow_html=True)
                        st.code(full_prompt_to_copy, language='markdown')
                else: st.info("该区间暂无单词")

        render_tab(t_top, top_df, "Top精选", expand_default=True, df_key="top") 
        render_tab(t_target, df[df['final_cat']=='target'], "重点", expand_default=False, df_key="target")
        render_tab(t_beyond, df[df['final_cat']=='beyond'], "超纲", expand_default=False, df_key="beyond")
        render_tab(t_known, df[df['final_cat']=='known'], "熟词", expand_default=False, df_key="known")