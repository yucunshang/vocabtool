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
# 0. 全局常量与配置
# ==========================================
PAGE_CONFIG = {"layout": "wide", "page_title": "Vocab Master Pro", "page_icon": "🚀"}
MAX_WORKERS = 5
CHUNK_SIZE = 30
MAX_WORDS_LIMIT = 300
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

try:
    import PyPDF2
    import docx
except ImportError:
    pass

# ==========================================
# 2. 数据与 NLP 初始化
# ==========================================
@st.cache_data(show_spinner=False)
def load_knowledge_base():
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
                pass
                
    return data['terms'], data['proper'], data['patch'], data['ambiguous']

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource(show_spinner="正在初始化 NLP 引擎...")
def setup_nltk():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_data_dir)
    
    required_packages = ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']
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
                    pass
setup_nltk()

def get_lemma(w):
    try:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict: return w.lower()
        for pos in ['ADJ', 'ADV', 'VERB', 'NOUN']:
            if pos in lemmas_dict: return lemmas_dict[pos][0]
        return list(lemmas_dict.values())[0][0]
    except Exception:
        return w.lower()

@st.cache_data(show_spinner=False)
def load_vocab():
    vocab = {}
    file_path = next((f for f in ["coca_cleaned.csv", "data.csv"] if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [str(c).strip().lower() for c in df.columns]
            w_col = next((c for c in df.columns if 'word' in c or '单词' in c), None)
            r_col = next((c for c in df.columns if 'rank' in c or '排序' in c), None)
            
            if w_col and r_col:
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
                df = df.sort_values(r_col, ascending=True).drop_duplicates(subset=[w_col], keep='first')
                vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except Exception as e:
            st.warning(f"⚠️ 词频表加载失败: {e}")

    for word, rank in BUILTIN_PATCH_VOCAB.items(): 
        vocab[word] = rank
        
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
    uploaded_file.seek(0)
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
            with zipfile.ZipFile(uploaded_file) as z:
                text_blocks = []
                for filename in z.namelist():
                    if filename.endswith(('.html', '.xhtml', '.htm', '.xml')):
                        try:
                            content = z.read(filename).decode('utf-8', errors='ignore')
                            clean_text = re.sub(r'<[^>]+>', ' ', content)
                            text_blocks.append(clean_text)
                        except: pass
                text_content = " ".join(text_blocks)
    except Exception as e:
        return f"⚠️ 文件解析失败: {str(e)}"
    return text_content

# ==========================================
# 4. Prompt 模板引擎 (已更新：动态对应格式)
# ==========================================
def get_base_prompt_template(export_type="CSV"):
    """
    根据选择的 export_type (CSV/TXT) 动态生成对应的 Prompt
    """
    if export_type == "CSV":
        format_rule = """4. 输出格式标准 (CSV 格式)
- 分隔符：严格使用英文逗号 (,) 分隔两列。
- 引用规则：由于内容包含逗号或换行，**每个字段必须严格使用双引号 ("...") 包裹**。
- 结构： "Front_Content","Back_Content" """
        example = """"run a business","to manage a company<br><br><em>He quit to run a business.</em><br><br>【词源】源自..." """
    else: # TXT (Tab 分隔)
        format_rule = """4. 输出格式标准 (TXT/Tab 格式)
- 分隔符：严格使用 **制表符 (Tab)** 分隔两列 (不要使用逗号)。
- 引用规则：不要使用引号包裹字段，除非内容中确实包含 Tab。
- 结构： Front_Content [TAB] Back_Content """
        # 注意：这里用 [TAB] 表示制表符，实际 Prompt 中需要明确
        example = """run a business	to manage a company<br><br><em>He quit to run a business.</em><br><br>【词源】源自..."""

    return f"""【角色设定】 你是一位精通词源学、认知心理学以及 Anki 算法的“英语词汇专家”。请严格遵守以下标准，处理我提供的单词列表：

1. 核心原则
- 含义拆分：若单词有多个常用义项，拆分为多条数据。
- 严禁堆砌：每张卡片只承载一个特定语境下的含义。

2. 卡片正面 (Column 1)
- 内容：提供自然的短语或搭配 (Phrase/Collocation)，而非单个孤立单词。
- 样式：纯文本。

3. 卡片背面 (Column 2 - 整合页)
- 背面信息必须全部合并在第二列，并使用 HTML 标签排版。
- 结构顺序：英文释义 <br><br> <em>例句</em> <br><br> 【词源/记忆法】中文解析

{format_rule}

5. 数据清洗
- 自动修正拼写错误；对缩写提供全称。

💡 最终输出示例 (严格模仿此格式)：
{example}

【系统绝对强制指令】
直接输出最终的数据代码，不要包含 ```csv 或 markdown 标记，不要回复任何客套话。"""

# ==========================================
# 5. 多核并发 API 引擎
# ==========================================
def _fetch_deepseek_chunk(batch_words, prompt_template, api_key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    full_prompt = f"{prompt_template}\n\n待处理单词列表：\n{', '.join(batch_words)}"
    
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
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 402: return "❌ ERROR_402_NO_BALANCE"
            elif resp.status_code == 401: return "❌ ERROR_401_INVALID_KEY"
            resp.raise_for_status()
            
            result = resp.json()['choices'][0]['message']['content'].strip()
            # 清洗 Markdown 标记
            if result.startswith("```"):
                lines = result.split('\n')
                if len(lines) > 1:
                    result = '\n'.join(lines[1:-1]).strip()
            return result
        except requests.exceptions.RequestException:
            if attempt == 2: return f"\n🚨 请求失败"
            time.sleep(2)
    return f"\n🚨 生成超时"

def call_deepseek_api_chunked(prompt_template, words, progress_bar, status_text):
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return "⚠️ 错误：未配置 DEEPSEEK_API_KEY"
    if not words: return "⚠️ 错误：没有单词"
    
    if len(words) > MAX_WORDS_LIMIT:
        st.warning(f"⚠️ 本次仅截取前 {MAX_WORDS_LIMIT} 个单词。")
        words = words[:MAX_WORDS_LIMIT]

    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    total_words = len(words)
    processed_count = 0
    results_ordered = [None] * len(chunks)
    
    status_text.markdown("🚀 **正在连接 DeepSeek...**")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(_fetch_deepseek_chunk, chunk, prompt_template, api_key): i 
                for i, chunk in enumerate(chunks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    res = future.result()
                    if "ERROR_" in res: return res
                    results_ordered[idx] = res
                except: results_ordered[idx] = ""
                
                processed_count += len(chunks[idx])
                progress_bar.progress(min(processed_count / total_words, 1.0))
                status_text.markdown(f"**⚡ 处理进度：** `{processed_count} / {total_words}`")
    except Exception as e:
        return f"❌ 异常: {str(e)}"

    return "\n".join([r for r in results_ordered if r])

# ==========================================
# 6. 分析引擎
# ==========================================
def analyze_words(unique_word_list):
    unique_items = [] 
    JUNK = {'s', 't', 'd', 'm', 'll', 've', 're'}
    for item in unique_word_list:
        if len(item) < 2 and item not in ['a', 'i']: continue
        if item in JUNK: continue
        
        rank = vocab_dict.get(item, 99999)
        if item in BUILTIN_TECHNICAL_TERMS:
            unique_items.append({"word": f"{item} ({BUILTIN_TECHNICAL_TERMS[item]})", "rank": rank if rank!=99999 else 15000})
        elif item in PROPER_NOUNS_DB or item in AMBIGUOUS_WORDS:
            unique_items.append({"word": PROPER_NOUNS_DB.get(item, item.title()), "rank": rank})
        elif rank != 99999:
            unique_items.append({"word": item, "rank": rank})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 7. UI 与主逻辑
# ==========================================
st.title("🚀 Vocab Master Pro - Stable V5.1")
st.markdown("💡 支持粘贴长文或上传文件，**格式化 Prompt 自动适配**。")

if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False

def clear_all():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False

# --- 参数栏 ---
st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: current_level = st.number_input("🎯 起始词汇量", 0, 30000, 7500, 500)
with c2: target_level = st.number_input("🎯 目标词汇量", 0, 30000, 15000, 500)
with c3: top_n = st.number_input("🔥 提取数量", 10, 500, 50, 10)
with c4: min_rank = st.number_input("📉 过滤前N高频词", 0, 20000, 3500, 500)
st.markdown("</div>", unsafe_allow_html=True)

# --- 输入栏 ---
col_in1, col_in2 = st.columns([3, 2])
with col_in1: raw_text = st.text_area("📥 文本", height=150, key="raw_input_text")
with col_in2: uploaded_file = st.file_uploader("📂 文件", type=["txt", "pdf", "docx", "epub"], key=f"uploader_{st.session_state.uploader_key}")

c_btn1, c_btn2 = st.columns([5, 1])
if c_btn1.button("🚀 开始分析", type="primary", use_container_width=True):
    with st.spinner("Processing..."):
        txt = raw_text
        if uploaded_file: txt += "\n" + extract_text_from_file(uploaded_file)
        
        if not txt.strip():
            st.warning("无有效文本")
        else:
            words = re.findall(r"[a-zA-Z']+", txt)
            lemmas = [get_lemma(w) for w in words]
            st.session_state.base_df = analyze_words(list(set([l.lower() for l in lemmas])))
            st.session_state.lemma_text = " ".join(lemmas)
            st.session_state.is_processed = True

if c_btn2.button("清空", use_container_width=True): clear_all()

st.divider()

# ==========================================
# 8. 结果展示
# ==========================================
if st.session_state.is_processed:
    df = st.session_state.base_df.copy()
    if not df.empty:
        df['cat'] = pd.cut(df['rank'], bins=[-1, current_level, target_level, 999999], labels=['known', 'target', 'beyond'])
        df = df.sort_values('rank')
        
        # 数据集定义
        datasets = {
            "🔥 Top精选": df[df['rank'] >= min_rank].head(top_n),
            "🟡 重点词": df[df['cat']=='target'],
            "🔴 超纲词": df[df['cat']=='beyond'],
            "🟢 已掌握": df[df['cat']=='known']
        }
        
        tabs = st.tabs(list(datasets.keys()) + ["原文"])
        
        for i, (label, sub_df) in enumerate(datasets.items()):
            with tabs[i]:
                if sub_df.empty:
                    st.info("暂无数据")
                    continue
                
                # 预览
                with st.expander(f"查看列表 ({len(sub_df)}词)", expanded=(i==0)):
                    st.code("\n".join(sub_df['word'].tolist()), language='text')

                # 生成区
                st.write("#### 🤖 AI 卡片生成")
                col_fmt, col_act = st.columns([1, 4])
                with col_fmt:
                    # 格式选择器
                    fmt_opt = st.radio("格式:", ["CSV", "TXT"], horizontal=True, key=f"fmt_{i}")
                    ext = "csv" if fmt_opt == "CSV" else "txt"
                    
                with col_act:
                    if st.button(f"⚡ 生成 {label} Anki卡片", key=f"gen_{i}"):
                        pure_words = sub_df['word'].tolist()
                        # 获取动态 Prompt
                        prompt = get_base_prompt_template(fmt_opt)
                        
                        pb = st.progress(0)
                        st_status = st.empty()
                        
                        res = call_deepseek_api_chunked(prompt, pure_words, pb, st_status)
                        
                        if "❌" in res:
                            st.error(res)
                        else:
                            st_status.success("完成！")
                            st.download_button(f"📥 下载 .{ext}", res, f"anki_{label}.{ext}", "text/plain", type="primary")
                            st.code(res, language="text" if fmt_opt=="TXT" else "csv")
                            
        with tabs[-1]:
            st.download_button("下载还原后全文", st.session_state.lemma_text, "lemmatized.txt")