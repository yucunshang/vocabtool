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
    pass  # 稍后在使用时提示，不阻断主程序启动

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
    .param-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据与 NLP 初始化 (健壮版)
# ==========================================
@st.cache_data
def load_knowledge_base():
    """加载本地知识库，具有容错能力"""
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
    except Exception as e:
        st.error(f"⚠️ 数据加载部分失败: {e}，将使用基础模式运行。")
    return data["terms"], data["proper"], data["patch"], data["ambiguous"]

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource
def setup_nltk():
    """NLTK 数据下载检查"""
    try:
        nltk.data.find('corpora/wordnet')
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
    """词形还原封装"""
    if not w: return ""
    try:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict: return w.lower()
        # 优先顺序：动词 -> 名词 -> 形容词 -> 副词
        for pos in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            if pos in lemmas_dict: return lemmas_dict[pos][0]
        return list(lemmas_dict.values())[0][0]
    except:
        return w.lower()

@st.cache_data
def load_vocab():
    """加载词频表，具备回退机制"""
    vocab = {}
    # 尝试加载 CSV
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
        except Exception:
            pass # 失败则依靠内置补丁
    
    # 合并补丁词库
    if BUILTIN_PATCH_VOCAB:
        for word, rank in BUILTIN_PATCH_VOCAB.items(): vocab[word] = rank
        
    # 紧急硬编码覆盖 (Common Overrides)
    URGENT_OVERRIDES = {
        "china": 400, "usa": 200, "uk": 200, "google": 1000, "apple": 1000, 
        "january": 400, "february": 400, "march": 400, "april": 400, "may": 100, "june": 400,
        "monday": 300, "sunday": 300
    }
    for word, rank in URGENT_OVERRIDES.items(): vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 3. 文档解析引擎
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
        return f" [解析错误: {e}] "
    return ""

def get_base_prompt_template(export_format="TXT"):
    return f"""【角色设定】 你是一位精通词源学、认知心理学以及 Anki 算法的“英语词汇专家与闪卡制作大师”。请严格遵守以下标准：

1. 核心原则：原子性 (Atomicity)
若一个单词有多个常用含义，必须拆分为多条独立数据。
2. 卡片正面 (Column 1)
提供自然的短语或搭配 (Phrase/Collocation)。
3. 卡片背面 (Column 2 - 整合页)
使用 HTML 标签排版，包含三个部分，用 <br><br> 分隔：
英文释义 <br><br> <em>斜体例句</em> <br><br> 【中文词源/记忆法】
4. 输出格式标准 ({export_format} 格式)
纯文本代码块，无 Markdown 包裹。
逗号分隔，字段用双引号包裹。
5. 数据清洗
自动修正拼写错误。

💡 最终输出示例（{export_format} 内容）：
"run a business","to manage a company<br><br><em>He quit his job to run a business.</em><br><br>【词源】run 源自古英语 rinnan（流动/运转）"
"""

# ==========================================
# 4. 多核并发 API 引擎 (线程安全版)
# ==========================================
def _fetch_deepseek_chunk_safe(batch_data):
    """
    纯函数，不操作 UI。
    batch_data: (index, words_list, prompt_template, api_key)
    Return: (index, result_string, error_msg)
    """
    index, batch_words, prompt_template, api_key = batch_data
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    system_enforcement = "\n\n【系统绝对强制指令】直接输出最终的数据代码，不要回复“好的”，不要使用 ```csv 包裹！"
    full_prompt = f"{prompt_template}{system_enforcement}\n\n待处理单词列表：\n{', '.join(batch_words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3,
        "max_tokens": 4096
    }
    
    try:
        for attempt in range(3):
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429: 
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 402: return (index, "", "ERROR_402_NO_BALANCE")
            elif resp.status_code == 401: return (index, "", "ERROR_401_INVALID_KEY")
            
            resp.raise_for_status()
            result = resp.json()['choices'][0]['message']['content'].strip()
            
            # 清洗 Markdown 标记
            if result.startswith("```"):
                lines = result.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                result = '\n'.join(lines).strip()
            return (index, result, None)
            
        return (index, "", "TIMEOUT")
    except Exception as e:
        return (index, "", str(e))

def run_concurrent_api(words, prompt_template, api_key, progress_bar, status_text):
    """主线程控制进度的并发器"""
    MAX_WORDS = 300 # 限制单次请求量
    if len(words) > MAX_WORDS:
        st.warning(f"⚠️ 为保证并发稳定，本次仅截取前 **{MAX_WORDS}** 个单词。")
        words = words[:MAX_WORDS]

    CHUNK_SIZE = 30
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    total_chunks = len(chunks)
    
    # 准备任务数据
    tasks = [(i, chunk, prompt_template, api_key) for i, chunk in enumerate(chunks)]
    results_map = {}
    errors = []
    
    status_text.markdown("🚀 **并发任务已发射！** 正在全速生成...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(_fetch_deepseek_chunk_safe, task): task[0] for task in tasks}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, res_str, err = future.result()
            
            if err:
                if "402" in err: return "❌ 错误：DeepSeek 账户余额不足。"
                if "401" in err: return "❌ 错误：API Key 无效。"
                errors.append(f"批次 {idx} 失败: {err}")
            else:
                results_map[idx] = res_str
            
            completed_count += 1
            progress = completed_count / total_chunks
            progress_bar.progress(progress)
            status_text.markdown(f"**⚡ AI 多核并发全速编纂中：** `{completed_count}/{total_chunks}` 批次完成")

    # 按原始顺序拼接
    final_output = []
    for i in range(total_chunks):
        if i in results_map:
            final_output.append(results_map[i])
    
    if errors:
        st.warning(f"⚠️ 部分批次生成失败 ({len(errors)}个)，已自动跳过。")
        
    return "\n".join(final_output)

# ==========================================
# 5. 分析引擎
# ==========================================
def analyze_words(unique_word_list):
    unique_items = [] 
    # 基础停用词过滤 (简单版)
    STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'}
    
    for item_lower in unique_word_list:
        if len(item_lower) < 2 and item_lower != 'a': continue
        if item_lower in STOP_WORDS: continue
        
        actual_rank = vocab_dict.get(item_lower, 99999)
        
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            unique_items.append({"word": f"{item_lower} ({domain})", "rank": term_rank, "raw": item_lower})
            continue
        
        # 简单过滤纯数字
        if item_lower.isdigit(): continue

        if actual_rank != 99999:
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower})
        elif item_lower in PROPER_NOUNS_DB: # 专有名词即使没排名也保留
             unique_items.append({"word": PROPER_NOUNS_DB[item_lower], "rank": 99999, "raw": item_lower})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 6. UI 与 状态管理
# ==========================================
st.title("🚀 Vocab Master Pro - V5")

# 初始化 Session State
if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False
if "generated_cards" not in st.session_state: st.session_state.generated_cards = {} # 用于存储 AI 生成结果

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False
    st.session_state.generated_cards = {} # 清空生成记录

# --- 顶部配置面板 (替代侧边栏) ---
with st.expander("⚙️ **参数配置与 API 设置** (点击展开)", expanded=False):
    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        # 尝试自动获取 Secrets
        default_key = ""
        try: default_key = st.secrets["DEEPSEEK_API_KEY"]
        except: pass
        user_api_key = st.text_input("🔑 DeepSeek API Key", value=default_key, type="password", help="如果没有配置 Secrets，请在此处输入")
    
    with col_k2:
        st.info("💡 参数说明：**忽略前 N 词** 可过滤掉 too, the 等简单词；**Top N** 选取最高频生词。")

    c1, c2, c3, c4 = st.columns(4)
    with c1: current_level = st.number_input("🎯 当前词汇量 (起)", 0, 30000, 4500, 500)
    with c2: target_level = st.number_input("🎯 目标词汇量 (止)", 0, 30000, 15000, 500)
    with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 50, 10)
    with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 1000, 500)
    
    show_rank = st.checkbox("🔢 在列表中显示词频 Rank", value=True)

# --- 输入区 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本 / 词表", height=150, key="raw_input_text", placeholder="在此粘贴英文文章、论文或单词列表...")
with col_input2:
    st.markdown("#### 📂 文档解析")
    uploaded_file = st.file_uploader("支持 TXT, PDF, DOCX, EPUB", type=["txt", "pdf", "docx", "epub"], key=f"uploader_{st.session_state.uploader_key}")

col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1: btn_process = st.button("🚀 开始极速解析", type="primary", use_container_width=True)
with col_btn2: st.button("🗑️ 清空", on_click=clear_all_inputs, use_container_width=True)

st.divider()

# ==========================================
# 7. 核心处理逻辑
# ==========================================
if btn_process:
    with st.spinner("🧠 正在进行文本清洗与词源分析..."):
        start_time = time.time()
        combined_text = raw_text
        if uploaded_file is not None: combined_text += "\n" + extract_text_from_file(uploaded_file)
            
        if not combined_text.strip():
            st.warning("⚠️ 未提取到任何有效文本！")
            st.session_state.is_processed = False
        else:
            # 优化正则：保留连字符单词 (state-of-the-art)
            raw_words = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", combined_text)
            
            # 词形还原
            lemmatized_words = [get_lemma(w) for w in raw_words]
            full_lemmatized_text = " ".join(lemmatized_words)
            unique_lemmas = list(set([w.lower() for w in lemmatized_words]))
            
            # 词频分析
            st.session_state.base_df = analyze_words(unique_lemmas)
            st.session_state.lemma_text = full_lemmatized_text
            st.session_state.stats = {
                "raw_count": len(raw_words),
                "unique_count": len(unique_lemmas),
                "valid_count": len(st.session_state.base_df),
                "time": time.time() - start_time
            }
            st.session_state.is_processed = True
            # 新分析时重置生成结果
            st.session_state.generated_cards = {} 

# ==========================================
# 8. 结果渲染
# ==========================================
if st.session_state.get("is_processed", False):
    
    stats = st.session_state.stats
    # 使用容器美化 Metrics
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📝 总词数", f"{stats['raw_count']:,}")
        c2.metric("✂️ 去重词汇", f"{stats['unique_count']:,}")
        c3.metric("🎯 有效学习词", f"{stats['valid_count']:,}")
        c4.metric("⚡ 耗时", f"{stats['time']:.2f}s")
    
    df = st.session_state.base_df.copy()
    
    if not df.empty:
        # 分级逻辑
        def categorize(row):
            r = row['rank']
            if r <= current_level: return "known"
            elif r <= target_level: return "target"
            else: return "beyond"
        
        df['final_cat'] = df.apply(categorize, axis=1)
        df = df.sort_values(by='rank')
        
        # 筛选逻辑
        top_df = df[df['rank'] >= min_rank_threshold].sort_values(by='rank', ascending=True).head(top_n)
        target_df = df[df['final_cat']=='target']
        beyond_df = df[df['final_cat']=='beyond']
        
        # Tabs
        tab_list = ["🔥 Top精选", "🟡 重点词汇", "🔴 超纲词汇", "📝 原文下载"]
        tabs = st.tabs(tab_list)
        
        # 渲染函数 (封装以复用)
        def render_word_tab(tab_obj, data_df, tab_key):
            with tab_obj:
                if data_df.empty:
                    st.info("该区间暂无单词")
                    return

                col_list, col_ai = st.columns([1, 2])
                
                # 左侧：单词列表
                with col_list:
                    st.markdown(f"**单词预览 ({len(data_df)})**")
                    display_text = []
                    for _, row in data_df.iterrows():
                        suffix = f" [{int(row['rank'])}]" if show_rank and row['rank'] != 99999 else ""
                        display_text.append(f"{row['word']}{suffix}")
                    st.text_area("列表", value="\n".join(display_text), height=400, label_visibility="collapsed")

                # 右侧：AI 生成区
                with col_ai:
                    st.markdown("#### 🤖 AI 卡片制作")
                    export_fmt = st.radio("格式", ["TXT", "CSV"], horizontal=True, key=f"fmt_{tab_key}")
                    
                    # 检查是否已有生成结果 (持久化)
                    res_key = f"{tab_key}_{export_fmt}"
                    existing_result = st.session_state.generated_cards.get(res_key)
                    
                    if existing_result:
                        st.success("✅ 卡片已生成！")
                        st.download_button(
                            label="📥 下载生成结果",
                            data=existing_result.encode('utf-8-sig'),
                            file_name=f"anki_{tab_key}.{export_fmt.lower()}",
                            mime="text/plain",
                            type="primary"
                        )
                        with st.expander("查看已生成内容"):
                            st.code(existing_result, language="text")
                    else:
                        st.info("点击下方按钮，调用 AI 为左侧单词生成解释、例句和词源。")
                        if st.button(f"⚡ 生成 {tab_key} 卡片", key=f"btn_{tab_key}"):
                            if not user_api_key:
                                st.error("❌ 请先在顶部配置栏输入 DeepSeek API Key")
                            else:
                                pure_words = data_df['word'].tolist()
                                prompt = get_base_prompt_template(export_fmt)
                                
                                p_bar = st.progress(0)
                                s_text = st.empty()
                                
                                result_str = run_concurrent_api(pure_words, prompt, user_api_key, p_bar, s_text)
                                
                                if "❌" in result_str and len(result_str) < 100:
                                    st.error(result_str)
                                else:
                                    # 保存结果到 Session State
                                    st.session_state.generated_cards[res_key] = result_str
                                    s_text.success("🎉 生成完成！")
                                    st.rerun() # 强制刷新以显示下载按钮

        render_word_tab(tabs[0], top_df, "top")
        render_word_tab(tabs[1], target_df, "target")
        render_word_tab(tabs[2], beyond_df, "beyond")
        
        with tabs[3]:
            st.download_button("💾 下载词形还原后的全文 (.txt)", st.session_state.lemma_text, "lemmatized.txt")
            st.text_area("全文预览", st.session_state.lemma_text[:5000], height=300)