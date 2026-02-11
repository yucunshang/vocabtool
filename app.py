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
from typing import Dict, List, Set, Tuple

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
    /* 自定义标签页样式 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #f9fafb; border-radius: 5px 5px 0 0; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 健壮的数据与 NLP 加载器
# ==========================================
@st.cache_data(show_spinner=False)
def load_knowledge_base():
    """带容错的知识库加载"""
    base_path = "data"
    default_res = ({}, {}, {}, set())
    if not os.path.exists(base_path):
        return default_res
    
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
    except Exception as e:
        st.warning(f"💡 部分知识库文件加载异常: {e}")
        return default_res

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource
def setup_nltk():
    """NLTK 初始化优化"""
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    for pkg in ['averaged_perceptron_tagger', 'punkt']:
        try:
            nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except Exception:
            pass

setup_nltk()

# 词形还原本地缓存，大幅提升长文处理速度
LEMMA_CACHE = {}

def get_lemma_optimized(w: str) -> str:
    w_lower = w.lower()
    if w_lower in LEMMA_CACHE:
        return LEMMA_CACHE[w_lower]
    
    lemmas_dict = lemminflect.getAllLemmas(w_lower)
    if not lemmas_dict:
        res = w_lower
    else:
        # 优先级排序：动词 > 形容词 > 名词
        res = w_lower
        for pos in ['VERB', 'ADJ', 'NOUN', 'ADV']:
            if pos in lemmas_dict:
                res = lemmas_dict[pos][0]
                break
        if res == w_lower:
            res = list(lemmas_dict.values())[0][0]
    
    LEMMA_CACHE[w_lower] = res
    return res

@st.cache_data
def load_vocab():
    """健壮的词频表加载逻辑"""
    vocab = {}
    # 尝试读取常见词频文件
    for f_name in ["coca_cleaned.csv", "data.csv", "data/coca.csv"]:
        if os.path.exists(f_name):
            try:
                df = pd.read_csv(f_name)
                df.columns = [str(c).strip().lower() for c in df.columns]
                w_col = next((c for c in df.columns if 'word' in c or '单词' in c), df.columns[0])
                r_col = next((c for c in df.columns if 'rank' in c or '排序' in c), df.columns[1])
                
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
                
                # 快速去重
                df = df.sort_values(r_col).drop_duplicates(subset=[w_col])
                vocab = dict(zip(df[w_col], df[r_col]))
                break
            except Exception:
                continue
    
    # 注入内置补丁
    vocab.update(BUILTIN_PATCH_VOCAB)
    # 注入高频专有名词/月份等
    OVERRIDES = {
        "china": 400, "google": 800, "apple": 800, "monday": 300, "january": 400, "usa": 200
    }
    vocab.update(OVERRIDES)
    return vocab

VOCAB_DICT = load_vocab()

# ==========================================
# 3. 文档解析引擎 (增强型)
# ==========================================
def extract_text_from_file(uploaded_file):
    """安全解析多格式文档"""
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        content = uploaded_file.read()
        if ext == 'txt':
            return content.decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            import PyPDF2
            from io import BytesIO
            reader = PyPDF2.PdfReader(BytesIO(content))
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            import docx
            from io import BytesIO
            doc = docx.Document(BytesIO(content))
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == 'epub':
            with zipfile.ZipFile(BytesIO(content)) as z:
                texts = []
                for f in z.namelist():
                    if f.endswith(('.html', '.xhtml', '.xml')):
                        raw = z.read(f).decode('utf-8', errors='ignore')
                        texts.append(re.sub(r'<[^>]+>', ' ', raw))
                return " ".join(texts)
    except ImportError as e:
        st.error(f"❌ 缺少必要依赖库: {str(e).split()[-1]}。请联系管理员安装。")
    except Exception as e:
        st.error(f"❌ 解析文件 {uploaded_file.name} 失败: {e}")
    return ""

# ==========================================
# 4. AI 调度引擎 (带指数退避)
# ==========================================
def _fetch_deepseek_chunk(batch_words: List[str], prompt_template: str, api_key: str):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    # 强制性输出指令优化：减少AI废话
    instruction = "\n\n[System Instruction: Output RAW text for CSV only. No markdown, no intros, no conversational fillers.]\n\nWords:\n"
    full_prompt = f"{prompt_template}{instruction}{', '.join(batch_words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.2, # 降低随机性
        "max_tokens": 4000
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1)) # 指数退避
                continue
            if resp.status_code == 402: return "❌ 账户余额不足"
            if resp.status_code == 401: return "❌ API KEY 无效"
            
            resp.raise_for_status()
            res_json = resp.json()
            content = res_json['choices'][0]['message']['content'].strip()
            
            # 清洗 Markdown 语法块
            content = re.sub(r'^```[a-zA-Z]*\n', '', content)
            content = re.sub(r'\n```$', '', content)
            return content
        except Exception as e:
            if attempt == 2: return f"❌ 批次请求失败: {str(e)}"
            time.sleep(2)
    return "❌ 未知错误"

def call_deepseek_api_chunked(prompt_template, words, progress_bar, status_text):
    """并发控制器"""
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "⚠️ 请在 Streamlit Secrets 中配置 DEEPSEEK_API_KEY"
    
    if not words: return ""

    # 动态批次大小：每批 25-30 词是 API 稳定性的黄金平衡点
    CHUNK_SIZE = 25
    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    results = [None] * len(chunks)
    
    # 使用 st.status 替换普通 text，Vibe 更高级
    with st.status("🚀 AI 并发引擎正在全速工作...", expanded=True) as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {executor.submit(_fetch_deepseek_chunk, chunks[i], prompt_template, api_key): i for i in range(len(chunks))}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                results[idx] = res
                completed += 1
                progress_bar.progress(completed / len(chunks))
                status.write(f"✅ 批次 {idx+1}/{len(chunks)} 完成 ({len(chunks[idx])} 词)")
        
        status.update(label="✨ 编纂任务全部完成！", state="complete", expanded=False)

    return "\n".join(filter(None, results))

# ==========================================
# 5. 分析流水线
# ==========================================
def process_pipeline(text: str):
    """高度优化的分析流水线"""
    if not text.strip(): return None, None
    
    # 1. 极速清洗与分词
    raw_words = re.findall(r"\b[a-zA-Z']{2,}\b", text) # 忽略单字母
    
    # 2. 统计原始频率以优化词形还原性能
    word_counts = Counter(raw_words)
    
    # 3. 批量词形还原 (使用本地缓存)
    unique_lemmas_map = {}
    for word in word_counts:
        lemma = get_lemma_optimized(word)
        unique_lemmas_map[lemma] = unique_lemmas_map.get(lemma, 0) + word_counts[word]
    
    # 4. 组装结果
    data = []
    for lemma, count in unique_lemmas_map.items():
        rank = VOCAB_DICT.get(lemma, 99999)
        
        display_name = lemma
        if lemma in BUILTIN_TECHNICAL_TERMS:
            display_name = f"{lemma} ({BUILTIN_TECHNICAL_TERMS[lemma]})"
        elif lemma in PROPER_NOUNS_DB:
            display_name = PROPER_NOUNS_DB[lemma]
            
        data.append({
            "word": display_name,
            "rank": rank,
            "count": count,
            "raw": lemma
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('rank', ascending=True)
        
    return df, raw_words

# ==========================================
# 6. Streamlit UI 交互
# ==========================================
st.title("🚀 Vocab Master Pro")
st.caption("大师级英语学习利器：智能分级、词形还原、多并发 AI 制卡")

# 初始化状态
if "is_processed" not in st.session_state: st.session_state.is_processed = False

# 参数配置区
with st.container():
    st.markdown("<div class='param-box'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: cur_lv = st.number_input("🎯 当前词频 (起)", 0, 20000, 3500)
    with c2: tgt_lv = st.number_input("🎯 目标词频 (止)", 0, 30000, 12000)
    with c3: top_n = st.number_input("🔥 精选 Top N", 5, 300, 50)
    with c4: 
        st.write("")
        show_rank = st.checkbox("显示词频 Rank", True)
    st.markdown("</div>", unsafe_allow_html=True)

# 输入区
col_in1, col_in2 = st.columns([2, 1])
with col_in1:
    raw_input = st.text_area("📥 输入文本", height=200, placeholder="在此粘贴长篇英文文章、论文或小说内容...")
with col_in2:
    uploaded_file = st.file_uploader("📂 文档解析", type=["txt", "pdf", "docx", "epub"])
    if st.button("🗑️ 清空所有输入", use_container_width=True):
        st.rerun()

if st.button("🚀 开始智能解析", type="primary", use_container_width=True):
    full_text = raw_input
    if uploaded_file:
        full_text += "\n" + extract_text_from_file(uploaded_file)
    
    if full_text.strip():
        with st.spinner("🧠 深度分析引擎运行中..."):
            start = time.time()
            df, raw_words = process_pipeline(full_text)
            if df is not None:
                st.session_state.df = df
                st.session_state.raw_count = len(raw_words)
                st.session_state.duration = time.time() - start
                st.session_state.is_processed = True
    else:
        st.warning("请先输入文本或上传文件。")

# 结果渲染
if st.session_state.get("is_processed"):
    df = st.session_state.df
    
    # 指标展示
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("词汇总量", f"{st.session_state.raw_count}")
    m2.metric("独立词根", f"{len(df)}")
    m3.metric("需重点学习", f"{len(df[(df['rank'] > cur_lv) & (df['rank'] <= tgt_lv)])}")
    m4.metric("解析耗时", f"{st.session_state.duration:.2f}s")
    
    # 分类逻辑
    target_df = df[(df['rank'] > cur_lv) & (df['rank'] <= tgt_lv)].copy()
    beyond_df = df[df['rank'] > tgt_lv].copy()
    known_df = df[df['rank'] <= cur_lv].copy()
    top_n_df = target_df.head(top_n)

    tabs = st.tabs([f"🔥 Top {len(top_n_df)}", "🟡 重点词", "🔴 超纲词", "🟢 已掌握", "📋 导出原文"])
    
    def render_vocab_tab(tab, data_df, key_prefix):
        with tab:
            if data_df.empty:
                st.info("该范围内没有单词。")
                return
            
            # 单词列表预览
            words_to_show = []
            for _, row in data_df.iterrows():
                label = f"{row['word']} [{int(row['rank'])}]" if show_rank and row['rank'] < 99999 else row['word']
                words_to_show.append(label)
            
            with st.expander("👁️ 查看待处理单词列表"):
                st.code("\n".join(words_to_show))
            
            st.divider()
            
            # AI 生成区
            st.subheader("🤖 AI 卡片自动构建")
            exp_fmt = st.radio("导出格式", ["TXT (Anki)", "CSV"], horizontal=True, key=f"fmt_{key_prefix}")
            
            at1, at2 = st.tabs(["⚡ 内置并发生成", "🔗 复制 Prompt 手动生成"])
            
            with at1:
                if st.button(f"✨ 召唤 AI 编纂 {len(data_df)} 个单词", key=f"btn_{key_prefix}"):
                    p_bar = st.progress(0)
                    from app import get_base_prompt_template # 假设原 template 函数保留
                    prompt = get_base_prompt_template(exp_fmt)
                    
                    raw_words_list = data_df['raw'].tolist()
                    result = call_deepseek_api_chunked(prompt, raw_words_list, p_bar, st.empty())
                    
                    if result and "❌" not in result:
                        st.download_button(
                            "📥 下载 Anki 导入文件", 
                            result.encode('utf-8-sig'), 
                            file_name=f"anki_{key_prefix}.{exp_fmt.lower()}",
                            mime="text/plain",
                            use_container_width=True,
                            type="primary"
                        )
                        st.code(result, language="text")
                    else:
                        st.error(result)
            
            with at2:
                from app import get_base_prompt_template
                full_p = f"{get_base_prompt_template(exp_fmt)}\n\nWords:\n{', '.join(data_df['raw'].tolist())}"
                st.code(full_p)

    render_vocab_tab(tabs[0], top_n_df, "top")
    render_vocab_tab(tabs[1], target_df, "target")
    render_vocab_tab(tabs[2], beyond_df, "beyond")
    render_vocab_tab(tabs[3], known_df, "known")
    
    with tabs[4]:
        st.write("此处可根据需要增加导出逻辑...")