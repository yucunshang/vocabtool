import streamlit as st
import pandas as pd
import re
import os
import json
import time
import requests
import zipfile
import concurrent.futures

# 文本处理库
import lemminflect
import nltk

# ==========================================
# 0. 全局常量与配置 (集中管理，方便维护)
# ==========================================
PAGE_CONFIG = {"layout": "wide", "page_title": "Vocab Master Pro", "page_icon": "🚀"}
MAX_WORKERS = 5         # API 并发线程数
CHUNK_SIZE = 30         # 每次请求的单词数量
MAX_WORDS_LIMIT = 300   # 限制单次最大处理词数 (防止 API 账单爆炸)
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

CUSTOM_CSS = """
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 16px !important; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    [data-testid="stMetricValue"] { font-size: 28px !important; color: var(--primary-color) !important; }
    .param-box { background-color: var(--secondary-background-color); padding: 15px 20px 5px 20px; border-radius: 10px; border: 1px solid var(--border-color-light); margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
</style>
"""

# ==========================================
# 1. 基础初始化
# ==========================================
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 尝试导入多格式文档处理库 (静默失败，仅在调用时报错)
try:
    import PyPDF2
    import docx
except ImportError:
    pass  # 延迟报错，避免启动时直接 Crash

# ==========================================
# 2. 数据与 NLP 初始化 (增强稳定性)
# ==========================================
@st.cache_data(show_spinner=False)
def load_knowledge_base():
    """加载本地 JSON 知识库，缺失时返回空对象以防报错"""
    base_path = 'data'
    data = {'terms': {}, 'proper': {}, 'patch': {}, 'ambiguous': set()}
    
    files_map = {
        'terms': ('terms.json', lambda x: {k.lower(): v for k, v in x.items()}),
        'proper': ('proper.json', lambda x: {k.lower(): v for k, v in x.items()}),
        'patch': ('patch.json', lambda x: x),
        'ambiguous': ('ambiguous.json', lambda x: set(x))
    }

    if not os.path.exists(base_path):
        return data['terms'], data['proper'], data['patch'], data['ambiguous']

    for key, (filename, processor) in files_map.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[key] = processor(json.load(f))
            except Exception:
                pass # 文件损坏或格式错误时跳过
                
    return data['terms'], data['proper'], data['patch'], data['ambiguous']

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource(show_spinner="正在初始化 NLP 引擎...")
def setup_nltk():
    """更稳健的 NLTK 初始化，避免重复下载和 SSL 错误"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_data_dir) # 优先使用本地目录
    
    required_packages = ['averaged_perceptron_tagger', 'punkt', 'punkt_tab'] # punkt_tab 兼容新版 NLTK
    for pkg in required_packages:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{pkg}')
            except LookupError:
                try:
                    nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
                except Exception:
                    pass # 网络不通时忽略，依靠 fallback
setup_nltk()

def get_lemma(w):
    """获取词元，增加异常保护"""
    try:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict: return w.lower()
        # 优先顺序: 形容词 -> 副词 -> 动词 -> 名词
        for pos in ['ADJ', 'ADV', 'VERB', 'NOUN']:
            if pos in lemmas_dict: return lemmas_dict[pos][0]
        return list(lemmas_dict.values())[0][0]
    except Exception:
        return w.lower()

@st.cache_data(show_spinner=False)
def load_vocab():
    """加载词频表，增加列名匹配的鲁棒性"""
    vocab = {}
    file_path = next((f for f in ["coca_cleaned.csv", "data.csv"] if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # 统一列名为小写并去除空格
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            # 模糊匹配列名
            w_col = next((c for c in df.columns if 'word' in c or '单词' in c), None)
            r_col = next((c for c in df.columns if 'rank' in c or '排序' in c), None)
            
            if w_col and r_col:
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
                df = df.sort_values(r_col, ascending=True).drop_duplicates(subset=[w_col], keep='first')
                vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except Exception as e:
            st.warning(f"⚠️ 词频表加载失败: {e}，将仅使用内置补丁数据。")

    # 合并补丁词库
    for word, rank in BUILTIN_PATCH_VOCAB.items(): 
        vocab[word] = rank
        
    # 紧急硬编码修正 (保持原逻辑)
    URGENT_OVERRIDES = {
        "china": 400, "turkey": 1500, "march": 500, "may": 100, "august": 1500, "polish": 2500,
        "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, "saturday": 300, "sunday": 300,
        "january": 400, "february": 400, "april": 400, "june": 400, "july": 400, "september": 400, "october": 400, "november": 400, "december": 400,
        "usa": 200, "uk": 200, "google": 1000, "apple": 1000, "microsoft": 1500
    }
    vocab.update(URGENT_OVERRIDES)
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 3. 文档解析引擎
# ==========================================
def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    uploaded_file.seek(0) # 确保指针在开头
    text_content = ""
    
    try:
        if ext == 'txt':
            text_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            if 'PyPDF2' not in globals(): return "⚠️ 缺少 PyPDF2 库"
            reader = PyPDF2.PdfReader(uploaded_file)
            text_content = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            if 'docx' not in globals(): return "⚠️ 缺少 python-docx 库"
            doc = docx.Document(uploaded_file)
            text_content = " ".join([p.text for p in doc.paragraphs])
        elif ext == 'epub':
            # EPUB 解析优化：使用 ZipFile 读取
            with zipfile.ZipFile(uploaded_file) as z:
                text_blocks = []
                for filename in z.namelist():
                    if filename.endswith(('.html', '.xhtml', '.htm', '.xml')):
                        try:
                            content = z.read(filename).decode('utf-8', errors='ignore')
                            # 简单的正则去标签，速度快
                            clean_text = re.sub(r'<[^>]+>', ' ', content)
                            text_blocks.append(clean_text)
                        except: pass
                text_content = " ".join(text_blocks)
    except Exception as e:
        return f"⚠️ 文件解析失败: {str(e)}"
    
    return text_content

def get_base_prompt_template(export_format="TXT"):
    return f"""【角色设定】 你是一位精通词源学、认知心理学以及 Anki 算法的“英语词汇专家与闪卡制作大师”。接下来的对话中，请严格遵守以下 5 项制卡标准，处理我提供的所有单词列表：

1. 核心原则：原子性 (Atomicity)
含义拆分：若一个单词有多个常用含义（名词 vs 动词，字面义 vs 引申义等），必须拆分为多条独立数据。
严禁堆砌：每张卡片只承载一个特定语境下的含义。
2. 卡片正面 (Column 1: Front)
内容：提供自然的短语或搭配 (Phrase/Collocation)，而非单个孤立单词。
样式：使用纯文本。
3. 卡片背面 (Column 2: Back - 整合页)
背面信息必须全部合并在第二列，并使用 HTML 标签排版：
英文释义：简练准确。
例句：使用 <em> 标签包裹。
【词根词缀】：用中文进行解析。
换行要求：三部分之间使用 <br><br> 分隔。
4. 输出格式标准 ({export_format} 格式)
文件规范：纯文本代码块。
分隔符：使用逗号 (Comma) 分隔字段。
引号包裹：每个字段必须用双引号 ("...") 包裹。
5. 数据清洗与优化
自动修正拼写错误；对缩写提供全称。

💡 最终输出示例：
"run a business","to manage or operate a company<br><br><em>He quit his job to run a business.</em><br><br>【词源】源自古英语 rinnan（跑/流动）"

导入提醒： 务必勾选 "Allow HTML in fields" (允许在字段中使用 HTML)。"""

# ==========================================
# 4. 多核并发 API 引擎 (核心极速区)
# ==========================================
def _fetch_deepseek_chunk(batch_words, prompt_template, api_key):
    """内部工作线程：负责单一批次的极速请求"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    system_enforcement = "\n\n【系统绝对强制指令】直接输出最终的数据代码，不要回复任何客套话，不要使用 Markdown 包裹！"
    full_prompt = f"{prompt_template}{system_enforcement}\n\n待处理单词列表：\n{', '.join(batch_words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
            
            if resp.status_code == 429: 
                time.sleep(2 * (attempt + 1)) # 指数退避
                continue
            if resp.status_code == 402: return "❌ ERROR_402_NO_BALANCE"
            elif resp.status_code == 401: return "❌ ERROR_401_INVALID_KEY"
            
            resp.raise_for_status()
            
            result = resp.json()['choices'][0]['message']['content'].strip()
            # 简单的 Markdown 清洗
            if result.startswith("```"):
                lines = result.split('\n')
                if len(lines) > 1:
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines[-1].startswith("```"): lines = lines[:-1]
                result = '\n'.join(lines).strip()
            return result
            
        except requests.exceptions.RequestException as e:
            if attempt == 2: return f"\n🚨 批次请求失败: {str(e)}"
            time.sleep(2)
            
    return f"\n🚨 批次超时 ({len(batch_words)}词) 生成失败。"

def call_deepseek_api_chunked(prompt_template, words, progress_bar, status_text):
    """多线程并发控制器"""
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return "⚠️ 错误：未配置 DEEPSEEK_API_KEY，请在 Streamlit Secrets 中添加。"
    
    if not words: return "⚠️ 错误：没有需要生成的单词。"
    
    # 限制最大处理量
    if len(words) > MAX_WORDS_LIMIT:
        st.warning(f"⚠️ 为保证并发稳定，本次仅截取前 **{MAX_WORDS_LIMIT}** 个单词。")
        words = words[:MAX_WORDS_LIMIT]

    # 切片
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    total_words = len(words)
    processed_count = 0
    results_ordered = [None] * len(chunks)
    
    status_text.markdown("🚀 **并发任务已发射！** 正在连接 DeepSeek 云端算力...")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(_fetch_deepseek_chunk, chunk, prompt_template, api_key): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                chunk_len = len(chunks[idx])
                
                try:
                    res = future.result()
                    if "ERROR_402" in res or "ERROR_401" in res: 
                        return res # 遇到鉴权/余额错误直接终止
                    results_ordered[idx] = res
                except Exception as e:
                    results_ordered[idx] = f"Error in chunk {idx}: {e}"
                
                processed_count += chunk_len
                current_progress = min(processed_count / total_words, 1.0)
                progress_bar.progress(current_progress)
                status_text.markdown(f"**⚡ AI 多核并发全速编纂中：** `{processed_count} / {total_words}` 词")
    except Exception as e:
        return f"❌ 线程池异常: {str(e)}"

    return "\n".join([r for r in results_ordered if r])

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
        
        # 1. 技术术语检查
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            unique_items.append({"word": f"{item_lower} ({domain})", "rank": term_rank, "raw": item_lower})
            continue
            
        # 2. 专有名词检查
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_lower.title())
            unique_items.append({"word": display, "rank": actual_rank, "raw": item_lower})
            continue
            
        # 3. 常规词汇
        if actual_rank != 99999:
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 6. UI 与流水线状态管理
# ==========================================
st.title("🚀 Vocab Master Pro - Stable V5")
st.markdown("💡 支持粘贴长文或上传 `TXT/PDF/DOCX/EPUB`，**内置 AI** 一键生成 Anki 卡片。")

if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False
    # 不需要 st.rerun()，Streamlit 按钮回调结束后会自动 rerun

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

# --- 输入区 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本", height=150, key="raw_input_text", placeholder="在此粘贴英语文章...")
with col_input2:
    st.info("💡 **多格式解析**：支持电子书/论文原著")
    uploaded_file = st.file_uploader("📂 上传文档", type=["txt", "pdf", "docx", "epub"], key=f"uploader_{st.session_state.uploader_key}")

col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1: btn_process = st.button("🚀 极速智能解析", type="primary", use_container_width=True)
with col_btn2: st.button("🗑️ 一键清空", on_click=clear_all_inputs, use_container_width=True)

st.divider()

# ==========================================
# 7. 逻辑处理核心
# ==========================================
if btn_process:
    with st.spinner("🧠 正在急速读取文件并进行智能解析..."):
        start_time = time.time()
        combined_text = raw_text
        if uploaded_file is not None: 
            file_text = extract_text_from_file(uploaded_file)
            combined_text += "\n" + file_text
            
        if not combined_text.strip():
            st.warning("⚠️ 未提取到任何有效文本！")
            st.session_state.is_processed = False
        else:
            # 文本预处理
            raw_words = re.findall(r"[a-zA-Z']+", combined_text)
            # 使用列表推导式加速
            lemmatized_words = [get_lemma(w) for w in raw_words]
            full_lemmatized_text = " ".join(lemmatized_words)
            
            unique_lemmas = list(set([w.lower() for w in lemmatized_words]))
            
            st.session_state.base_df = analyze_words(unique_lemmas)
            st.session_state.lemma_text = full_lemmatized_text
            st.session_state.stats = {
                "raw_count": len(raw_words),
                "unique_count": len(unique_lemmas),
                "valid_count": len(st.session_state.base_df),
                "time": time.time() - start_time
            }
            st.session_state.is_processed = True

# ==========================================
# 8. 结果渲染
# ==========================================
if st.session_state.get("is_processed", False):
    
    stats = st.session_state.stats
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("📝 解析总字数", f"{stats['raw_count']:,}")
    col_m2.metric("✂️ 去重词根数", f"{stats['unique_count']:,}")
    col_m3.metric("🎯 纳入分级词汇", f"{stats['valid_count']:,}")
    col_m4.metric("⚡ 耗时", f"{stats['time']:.2f} s")
    
    df = st.session_state.base_df.copy()
    
    if not df.empty:
        # 分组逻辑
        df['final_cat'] = pd.cut(
            df['rank'], 
            bins=[-1, current_level, target_level, 999999], 
            labels=['known', 'target', 'beyond']
        )
        
        # 排序与TopN
        df = df.sort_values(by='rank')
        top_df = df[df['rank'] >= min_rank_threshold].sort_values(by='rank').head(top_n)
        
        # 选项卡
        tabs = st.tabs([
            f"🔥 Top {len(top_df)}", 
            f"🟡 重点 ({len(df[df['final_cat']=='target'])})", 
            f"🔴 超纲 ({len(df[df['final_cat']=='beyond'])})", 
            f"🟢 已掌握 ({len(df[df['final_cat']=='known'])})",
            "📝 原文下载"
        ])
        
        def render_tab(tab_obj, data_df, label, expand_default=False, key_suffix=""):
            with tab_obj:
                if data_df.empty:
                    st.info("该区间暂无单词")
                    return

                pure_words = data_df['word'].tolist()
                
                # 预览区域
                with st.expander("👁️ 查看单词列表", expanded=expand_default):
                    display_list = [
                        f"{row['word']} [Rank: {int(row['rank'])}]" if show_rank and row['rank']!=99999 else row['word']
                        for _, row in data_df.iterrows()
                    ]
                    st.markdown("<p class='copy-hint'>👆 点击右上角复制</p>", unsafe_allow_html=True)
                    st.code("\n".join(display_list), language='text')
                
                st.divider()
                
                # AI 生成区
                export_format = st.radio("⚙️ 格式:", ["TXT", "CSV"], horizontal=True, key=f"fmt_{key_suffix}")
                ai_tab1, ai_tab2 = st.tabs(["🤖 内置 AI 生成", "📋 复制 Prompt"])
                
                with ai_tab1:
                    prompt_val = get_base_prompt_template(export_format)
                    custom_prompt = st.text_area("Prompt", value=prompt_val, height=150, key=f"p_{key_suffix}")
                    
                    if st.button(f"⚡ 生成 {label} 卡片", key=f"btn_{key_suffix}", type="primary"):
                        pb = st.progress(0)
                        st_txt = st.empty()
                        
                        start_t = time.time()
                        res = call_deepseek_api_chunked(custom_prompt, pure_words, pb, st_txt)
                        end_t = time.time()
                        
                        if "❌" in res and len(res) < 100:
                            st.error(res)
                        else:
                            st_txt.success(f"🎉 完成！耗时 {end_t - start_t:.2f}s")
                            ext = export_format.lower()
                            mime = "text/csv" if export_format == "CSV" else "text/plain"
                            st.download_button(f"📥 下载 .{ext}", res, f"anki_{label}.{ext}", mime, type="primary")
                            st.code(res, language="text")

                with ai_tab2:
                    full_p = f"{get_base_prompt_template(export_format)}\n\n待处理单词：\n{', '.join(pure_words)}"
                    st.code(full_p, language='markdown')

        render_tab(tabs[0], top_df, "Top精选", True, "top") 
        render_tab(tabs[1], df[df['final_cat']=='target'], "重点", False, "target")
        render_tab(tabs[2], df[df['final_cat']=='beyond'], "超纲", False, "beyond")
        render_tab(tabs[3], df[df['final_cat']=='known'], "熟词", False, "known")
        
        with tabs[4]:
            st.download_button("💾 下载词形还原全文", st.session_state.lemma_text, "lemmatized.txt")
            st.code(st.session_state.lemma_text[:2000] + "\n...(略)...", language='text')