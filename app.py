import streamlit as st
import pandas as pd
import re
import os
import json
import time
import requests
import zipfile
import concurrent.futures
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Any, Optional

# ==========================================
# 0. 依赖检查与导入
# ==========================================
try:
    import lemminflect
    import nltk
except ImportError:
    st.error("⚠️ 缺少核心 NLP 依赖。请运行: pip install lemminflect nltk")
    st.stop()

try:
    import PyPDF2
    import docx
except ImportError:
    st.warning("⚠️ 缺少文档处理依赖 (PyPDF2, python-docx)。PDF 和 DOCX 解析将不可用。")

# ==========================================
# 1. 基础配置 & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 16px !important; }
    header {visibility: hidden;} 
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    [data-testid="stMetricValue"] { font-size: 28px !important; color: var(--primary-color) !important; }
    .param-box { background-color: var(--secondary-background-color); padding: 15px 20px 5px 20px; border-radius: 10px; border: 1px solid var(--border-color-light); margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
    /* 进度条样式优化 */
    .stProgress > div > div > div > div { background-color: #00CC96; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 资源管理与 NLP 引擎
# ==========================================
DATA_DIR = Path("data")
NLTK_DIR = Path(__file__).parent / "nltk_data"

@st.cache_resource
def setup_nltk():
    """初始化 NLTK，确保数据存在"""
    os.makedirs(NLTK_DIR, exist_ok=True)
    nltk.data.path.append(str(NLTK_DIR))
    for pkg in ['averaged_perceptron_tagger', 'punkt']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            try:
                nltk.download(pkg, download_dir=str(NLTK_DIR), quiet=True)
            except Exception as e:
                st.warning(f"NLTK {pkg} 下载失败: {e}")

setup_nltk()

@st.cache_data
def load_knowledge_base() -> Tuple[Dict, Dict, Dict, Set]:
    """加载本地 JSON 知识库，具备容错能力"""
    def load_json_safe(filename: str, default_val: Any) -> Any:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            return default_val
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default_val

    terms = {k.lower(): v for k, v in load_json_safe('terms.json', {}).items()}
    proper = {k.lower(): v for k, v in load_json_safe('proper.json', {}).items()}
    patch = load_json_safe('patch.json', {})
    ambiguous = set(load_json_safe('ambiguous.json', []))
    
    return terms, proper, patch, ambiguous

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_data
def load_vocab() -> Dict[str, int]:
    """加载词频表 (Rank)"""
    vocab = {}
    # 优先查找存在的 CSV 文件
    possible_files = ["coca_cleaned.csv", "data.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # 更加鲁棒的列名匹配
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            
            # 动态寻找 word 和 rank 列
            w_col = next((c for c in cols if 'word' in c or '单词' in c), None)
            r_col = next((c for c in cols if 'rank' in c or '排序' in c), None)
            
            if w_col and r_col:
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
                df = df.sort_values(r_col, ascending=True).drop_duplicates(subset=[w_col], keep='first')
                vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except Exception as e:
            st.error(f"词频表加载异常: {e}")

    # 合并补丁词库
    for word, rank in BUILTIN_PATCH_VOCAB.items():
        vocab[word] = rank
        
    # 紧急硬编码覆盖 (Urgent Overrides)
    URGENT_OVERRIDES = {
        "china": 400, "turkey": 1500, "march": 500, "may": 100, "august": 1500, "polish": 2500,
        "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, 
        "saturday": 300, "sunday": 300, "january": 400, "february": 400, "april": 400, 
        "june": 400, "july": 400, "september": 400, "october": 400, "november": 400, 
        "december": 400, "usa": 200, "uk": 200, "google": 1000, "apple": 1000, "microsoft": 1500
    }
    vocab.update(URGENT_OVERRIDES)
    return vocab

vocab_dict = load_vocab()

# ⚡ 性能优化: LRU 缓存避免重复计算
@lru_cache(maxsize=10000)
def get_lemma(w: str) -> str:
    """获取单词的词元 (Lemma)，带缓存"""
    lemmas_dict = lemminflect.getAllLemmas(w)
    if not lemmas_dict:
        return w.lower()
    # 优先顺序: 形容词 > 副词 > 动词 > 名词 (根据经验调整)
    for pos in ['ADJ', 'ADV', 'VERB', 'NOUN']:
        if pos in lemmas_dict:
            return lemmas_dict[pos][0]
    return list(lemmas_dict.values())[0][0]

# ==========================================
# 3. 文档解析引擎
# ==========================================
def extract_text_from_file(uploaded_file) -> str:
    """从不同格式文件中提取文本"""
    if uploaded_file is None:
        return ""
        
    ext = uploaded_file.name.split('.')[-1].lower()
    uploaded_file.seek(0)
    
    try:
        if ext == 'txt':
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        
        elif ext == 'pdf':
            if 'PyPDF2' not in globals(): return "⚠️ 缺少 PyPDF2 库"
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() or "" for page in reader.pages])
            
        elif ext == 'docx':
            if 'docx' not in globals(): return "⚠️ 缺少 python-docx 库"
            doc = docx.Document(uploaded_file)
            return " ".join([p.text for p in doc.paragraphs])
            
        elif ext == 'epub':
            text_blocks = []
            with zipfile.ZipFile(uploaded_file) as z:
                for filename in z.namelist():
                    if filename.endswith(('.html', '.xhtml', '.htm', '.xml')):
                        try:
                            content = z.read(filename).decode('utf-8', errors='ignore')
                            # 简单的正则去除 HTML 标签
                            clean_text = re.sub(r'<[^>]+>', ' ', content)
                            text_blocks.append(clean_text)
                        except: pass
            return " ".join(text_blocks)
            
    except Exception as e:
        st.error(f"文件解析失败 ({ext}): {e}")
        return ""
    return ""

# ==========================================
# 4. API 引擎 (健壮并发版)
# ==========================================
def get_base_prompt_template(export_format="TXT"):
    return f"""【角色设定】 你是一位精通词源学、认知心理学以及 Anki 算法的“英语词汇专家与闪卡制作大师”。接下来的对话中，请严格遵守以下 5 项制卡标准，处理我提供的所有单词列表：

1. 核心原则：原子性 (Atomicity)
含义拆分：若一个单词有多个常用含义（名词 vs 动词，字面义 vs 引申义等），必须拆分为多条独立数据。
严禁堆砌：每张卡片只承载一个特定语境下的含义，不准将多个释义挤在一起。
2. 卡片正面 (Column 1: Front)
内容：提供自然的短语或搭配 (Phrase/Collocation)，而非单个孤立单词。
3. 卡片背面 (Column 2: Back - 整合页)
背面信息必须全部合并在第二列，并使用 HTML 标签排版。
结构示例：英文释义<br><br><em>斜体例句</em><br><br>【词根、词源、词缀】的中文解析
4. 输出格式标准 ({export_format} 格式)
文件规范：纯文本代码块。
分隔符：使用逗号 (Comma) 分隔字段。
引号包裹：每个字段必须用双引号 ("...") 包裹。
5. 数据清洗与优化
拼写修正：自动修正用户列表中的明显拼写错误。
缩写展开：对缩写（如 WFH, aka）在背面提供全称及解释。
💡 最终输出示例（{export_format} 内容）：
"run a business","to manage or operate a company<br><br><em>He quit his job to run a business selling handmade crafts.</em><br><br>【词源】源自古英语 rinnan（跑/流动），引申为“使机器运转”或“使业务流转”"
"""

def _fetch_deepseek_chunk_safe(batch_words: List[str], prompt_template: str, api_key: str) -> str:
    """Worker 函数：处理单个批次，不包含任何 UI 操作"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    system_enforcement = "\n\n【系统绝对强制指令】现在我已经发送了单词列表，请立即且直接输出最终的数据代码，绝对不准回复“好的”、“没问题”等任何客套话，绝对不准使用 ```csv 等 Markdown 语法包裹代码！"
    full_prompt = f"{prompt_template}{system_enforcement}\n\n待处理单词列表：\n{', '.join(batch_words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3,
        "max_tokens": 4096
    }
    
    # 指数退避重试策略
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429: # 限流
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 402: return "ERROR: NO_BALANCE"
            if resp.status_code == 401: return "ERROR: INVALID_KEY"
            
            resp.raise_for_status()
            
            result = resp.json()['choices'][0]['message']['content'].strip()
            
            # 清洗 Markdown 标记
            if result.startswith("```"):
                lines = result.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                result = '\n'.join(lines).strip()
            return result
            
        except requests.exceptions.RequestException as e:
            if attempt == 2: return f"ERROR: Request failed: {str(e)}"
            time.sleep(1)
            
    return "ERROR: Timeout"

def call_deepseek_api_main_thread_managed(prompt_template, words, progress_bar, status_container):
    """主线程管理的并发控制器"""
    try: 
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except (KeyError, FileNotFoundError):
        return "⚠️ 未配置 DEEPSEEK_API_KEY，请在 .streamlit/secrets.toml 中配置。"
    
    if not words: return "⚠️ 没有需要生成的单词。"
    
    # 限制单次请求量
    MAX_WORDS = 300
    if len(words) > MAX_WORDS:
        st.toast(f"⚠️ 单词数量过多，已截取前 {MAX_WORDS} 个单词进行处理。", icon="✂️")
        words = words[:MAX_WORDS]

    CHUNK_SIZE = 30  
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    total_chunks = len(chunks)
    
    results_map = {} # {index: result_text}
    
    # 使用 st.status 提供更好的 UI 反馈
    with status_container.status("🤖 AI 正在并发思考中...", expanded=True) as status:
        st.write("🚀 正在初始化并发线程池...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交任务，记录 index 以便最后按顺序重组
            future_to_idx = {
                executor.submit(_fetch_deepseek_chunk_safe, chunk, prompt_template, api_key): i 
                for i, chunk in enumerate(chunks)
            }
            
            completed_count = 0
            
            # as_completed 在主线程迭代，可以安全更新 UI
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results_map[idx] = res
                    
                    if "ERROR:" in res:
                        st.write(f"⚠️ 批次 {idx+1} 出现异常: {res}")
                    
                    completed_count += 1
                    progress = completed_count / total_chunks
                    progress_bar.progress(progress)
                    st.write(f"✅ 完成批次 {idx+1}/{total_chunks} ({len(chunks[idx])} 词)")
                    
                except Exception as exc:
                    st.error(f"❌ 批次 {idx+1} 严重崩溃: {exc}")
                    results_map[idx] = ""

        status.update(label="🎉 生成完成！", state="complete", expanded=False)

    # 按原始顺序重组结果
    final_output = []
    for i in range(total_chunks):
        if i in results_map and not results_map[i].startswith("ERROR"):
            final_output.append(results_map[i])
        elif i in results_map:
            final_output.append(f"<!-- Batch {i+1} Failed: {results_map[i]} -->")
            
    return "\n".join(final_output)

# ==========================================
# 5. 分析引擎
# ==========================================
def analyze_words(unique_word_list):
    unique_items = [] 
    JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're', 'don', 'isn', 'aren'} # 扩展垃圾词
    
    for item_lower in unique_word_list:
        # 基础过滤
        if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
        if item_lower in JUNK_WORDS: continue
        if item_lower.isdigit(): continue # 过滤纯数字
        
        actual_rank = vocab_dict.get(item_lower, 99999)
        
        # 1. 技术术语检测
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            unique_items.append({"word": f"{item_lower} ({domain})", "rank": term_rank, "raw": item_lower})
            continue
            
        # 2. 专有名词检测
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_lower.title())
            unique_items.append({"word": display, "rank": actual_rank, "raw": item_lower})
            continue
            
        # 3. 常规词汇
        if actual_rank != 99999:
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 6. UI 与 状态管理
# ==========================================
# 状态初始化
if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None # 存储分析结果 DataFrame
if "lemma_text" not in st.session_state: st.session_state.lemma_text = ""

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.analysis_result = None
    st.session_state.lemma_text = ""

# --- 参数配置区 ---
st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: current_level = st.number_input("🎯 当前词汇量 (起)", 0, 30000, 7500, 500, help="过滤掉过于简单的词")
with c2: target_level = st.number_input("🎯 目标词汇量 (止)", 0, 30000, 15000, 500, help="过滤掉过于生僻的词")
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 50, 10)
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 3500, 500, help="即便是重点词，如果太常见也不显示")
with c5: 
    st.write("") 
    st.write("") 
    show_rank = st.checkbox("🔢 附加显示 Rank", value=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- 输入区 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text_input = st.text_area("📥 粘贴文本", height=150, key="raw_input_text", placeholder="在此粘贴英文文章...")
with col_input2:
    uploaded_file = st.file_uploader("📂 上传文档 (TXT/PDF/DOCX/EPUB)", 
                                     type=["txt", "pdf", "docx", "epub"], 
                                     key=f"uploader_{st.session_state.uploader_key}")

col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1: btn_process = st.button("🚀 极速智能解析", type="primary", use_container_width=True)
with col_btn2: st.button("🗑️ 清空", on_click=clear_all_inputs, use_container_width=True)

st.divider()

# ==========================================
# 7. 逻辑处理流
# ==========================================
# 触发处理逻辑
if btn_process:
    with st.status("🧠 正在进行深度解析...", expanded=True) as status:
        start_time = time.time()
        
        # 1. 文本提取
        combined_text = raw_text_input
        if uploaded_file is not None: 
            status.write("📄 正在读取文件内容...")
            extracted = extract_text_from_file(uploaded_file)
            if extracted:
                combined_text += "\n" + extracted
            else:
                st.error("无法从文件中提取文本。")
        
        if not combined_text.strip():
            status.update(label="⚠️ 未检测到有效文本", state="error")
        elif not vocab_dict:
             status.update(label="⚠️ 词库未加载", state="error")
        else:
            # 2. NLP 处理
            status.write("🔍 正在分词与词形还原 (Lemmatization)...")
            # 预编译正则提高效率
            word_pattern = re.compile(r"[a-zA-Z']+")
            raw_words = word_pattern.findall(combined_text)
            
            # 使用带缓存的函数
            lemmatized_words = [get_lemma(w) for w in raw_words]
            unique_lemmas = list(set([w.lower() for w in lemmatized_words]))
            
            status.write(f"📊 正在比对 {len(unique_lemmas)} 个去重词汇...")
            analyzed_df = analyze_words(unique_lemmas)
            
            # 存入 Session State
            st.session_state.analysis_result = analyzed_df
            st.session_state.lemma_text = " ".join(lemmatized_words)
            st.session_state.stats = {
                "raw_count": len(raw_words),
                "unique_count": len(unique_lemmas),
                "valid_count": len(analyzed_df),
                "time": time.time() - start_time
            }
            status.update(label="✅ 解析完成！", state="complete", expanded=False)

# ==========================================
# 8. 结果渲染 (基于 Session State)
# ==========================================
if st.session_state.analysis_result is not None:
    stats = st.session_state.stats
    df = st.session_state.analysis_result.copy()
    
    # 顶部指标
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("📝 解析总字数", f"{stats['raw_count']:,}")
    col_m2.metric("✂️ 去重词根", f"{stats['unique_count']:,}")
    col_m3.metric("🎯 命中词库", f"{stats['valid_count']:,}")
    col_m4.metric("⚡ 耗时", f"{stats['time']:.2f} s")
    
    if not df.empty:
        # 动态分类逻辑 (在这里执行，这样修改 Slider 不需要重新解析文本)
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
        known_df = df[df['final_cat']=='known']
        
        t_top, t_target, t_beyond, t_known, t_raw = st.tabs([
            f"🔥 Top {len(top_df)}", 
            f"🟡 重点 ({len(target_df)})", 
            f"🔴 超纲 ({len(beyond_df)})", 
            f"🟢 已掌握 ({len(known_df)})",
            "📝 原文下载"
        ])
        
        def render_word_tab(tab_obj, data_df, label, key_prefix):
            with tab_obj:
                if data_df.empty:
                    st.info("此区间暂无数据")
                    return

                # 展示列表
                pure_words = data_df['word'].tolist()
                display_lines = [
                    f"{row['word']} [Rank: {int(row['rank'])}]" if show_rank else row['word']
                    for _, row in data_df.iterrows()
                ]
                
                with st.expander("👁️ 查看单词列表", expanded=(label=="Top")):
                    st.code("\n".join(display_lines), language='text')

                st.divider()
                
                # AI 生成区
                c_fmt, c_act = st.columns([1, 3])
                with c_fmt:
                    export_format = st.radio("输出格式:", ["TXT", "CSV"], horizontal=True, key=f"fmt_{key_prefix}")
                
                ai_tab1, ai_tab2 = st.tabs(["🤖 内置 AI 生成", "📋 复制 Prompt"])
                
                with ai_tab1:
                    custom_prompt = st.text_area("AI 指令模板", value=get_base_prompt_template(export_format), height=300, key=f"p_{key_prefix}")
                    if st.button(f"⚡ 生成 {label} 卡片", key=f"btn_{key_prefix}", type="primary"):
                        progress = st.progress(0)
                        status_box = st.empty()
                        
                        result_text = call_deepseek_api_main_thread_managed(
                            custom_prompt, pure_words, progress, status_box
                        )
                        
                        if result_text and "ERROR" not in result_text[:20]:
                            mime = "text/csv" if export_format == "CSV" else "text/plain"
                            st.download_button(
                                "📥 下载结果文件", 
                                data=result_text.encode('utf-8-sig'), 
                                file_name=f"anki_{label}_{int(time.time())}.{export_format.lower()}", 
                                mime=mime, 
                                type="primary"
                            )
                            with st.expander("预览结果"):
                                st.text(result_text)
                        elif "ERROR" in result_text:
                            st.error(result_text)

                with ai_tab2:
                    prompt_txt = f"{get_base_prompt_template(export_format)}\n\n单词列表:\n{', '.join(pure_words)}"
                    st.code(prompt_txt, language='markdown')

        render_word_tab(t_top, top_df, "Top", "top")
        render_word_tab(t_target, target_df, "Target", "target")
        render_word_tab(t_beyond, beyond_df, "Beyond", "beyond")
        render_word_tab(t_known, known_df, "Known", "known")
        
        with t_raw:
            st.info("含词形还原后的全文 (例如 'running' -> 'run')")
            st.download_button("💾 下载全文", st.session_state.lemma_text, "full_text.txt")
            st.text_area("预览", st.session_state.lemma_text[:2000] + "...", height=300)