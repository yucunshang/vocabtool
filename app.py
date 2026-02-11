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
        with open('data/terms.json', 'r', encoding='utf-8') as f: terms = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/proper.json', 'r', encoding='utf-8') as f: proper = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/patch.json', 'r', encoding='utf-8') as f: patch = json.load(f)
        with open('data/ambiguous.json', 'r', encoding='utf-8') as f: ambiguous = set(json.load(f))
        return terms, proper, patch, ambiguous
    except FileNotFoundError:
        st.error("⚠️ 缺少 data/ 文件夹下的 JSON 知识库文件！")
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
# 3. 文档解析 & AI 接口 & 提示词引擎
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

def get_base_prompt_template(export_format="CSV"):
    """经过 Anki 严格优化的防报错指令模板"""
    return f"""请扮演一位专业的 Anki 制卡专家。请严格按以下标准，为我提供直接可导入 Anki 的 {export_format} 格式数据。

核心原则与输出规范：
1. 结构强制：每行代表一张卡片，严格包含两个字段：正面,背面。
2. 分隔符：两个字段之间必须使用英文逗号 (,) 分隔。
3. 引号包裹：每个字段的内容必须使用英文双引号 ("...") 包裹。严禁在内容中使用未转义的双引号。
4. 卡片正面（字段1）：提供单词的自然搭配或短语。
5. 卡片背面（字段2 - HTML排版）：包含三个部分，必须使用 <br><br> 分隔：
   - 英文释义
   - <em>斜体例句</em>
   - 【词根/助记】中文解析

💡 最终输出格式示例：
"run a business","to manage a company<br><br><em>He runs a business.</em><br><br>【助记】源自古英语"
"go for a run","an act of running<br><br><em>I go for a run.</em><br><br>【助记】名词用法"

⚠️ 极其重要的格式警告：
绝对不要输出 Markdown 代码块标记（严禁使用 ```csv 或 ```txt ），不要有任何解释性的开场白或结束语！请直接输出纯文本数据本身！"""

def call_deepseek_api(prompt_template, words):
    try: api_key = st.secrets["DEEPSEEK_API_KEY"]
    except KeyError: return "⚠️ 站长配置错误：未在 Streamlit 后台 Secrets 中配置 DEEPSEEK_API_KEY。"
    if not words: return "⚠️ 错误：没有需要生成的单词。"
    url = "https://api.deepseek.com/chat/completions".strip()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    full_prompt = f"{prompt_template}\n\n待处理单词：\n{', '.join(words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3
    }
    
    try:
        # 设置超时时间，捕获潜在网络异常
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if resp.status_code == 402: return "❌ 错误：DeepSeek 账户余额不足，请充值。"
        elif resp.status_code == 401: return "❌ 错误：API Key 无效。"
        
        resp.raise_for_status()
        
        result = resp.json()['choices'][0]['message']['content']
        
        # 二次清洗：强行去除 AI 可能残留的 markdown 代码块外壳
        result = re.sub(r"^```(?:csv|txt|text)?\n", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\n```$", "", result)
        
        return result.strip()
    except requests.exceptions.Timeout:
        return "⏳ 请求超时：请稍后重试。"
    except Exception as e:
        return f"🚨 API 调用失败: {str(e)}"

# ==========================================
# 4. 分析引擎
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
# 5. UI 与流水线状态管理
# ==========================================
st.title("🚀 Vocab Master Pro - 全能教研引擎")
st.markdown("💡 支持粘贴长文或直接上传 `TXT / PDF / DOCX / EPUB` 原著电子书，并**内置免费 AI** 一键生成 Anki 记忆卡片。")

if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_processed" not in st.session_state: st.session_state.is_processed = False

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 
    st.session_state.is_processed = False

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
# 6. 后台硬核计算
# ==========================================
if btn_process:
    with st.spinner("🧠 正在急速读取文件并进行智能解析（长篇巨著请稍候）..."):
        start_time = time.time()
        combined_text = raw_text
        if uploaded_file is not None: combined_text += "\n" + extract_text_from_file(uploaded_file)
            
        if not combined_text.strip():
            st.warning("⚠️ 未提取到任何有效文本！")
            st.session_state.is_processed = False
        elif vocab_dict:
            raw_words = re.findall(r"[a-zA-Z']+", combined_text)
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
# 7. 动态界面渲染
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
        
        t_top, t_target, t_beyond, t_known, t_raw = st.tabs([
            f"🔥 Top {len(top_df)}", f"🟡 重点 ({len(df[df['final_cat']=='target'])})", 
            f"🔴 超纲 ({len(df[df['final_cat']=='beyond'])})", f"🟢 已掌握 ({len(df[df['final_cat']=='known'])})",
            "📝 原文防卡死下载"
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
                    
                    export_format = st.radio("⚙️ 选择输出格式:", ["CSV", "TXT"], horizontal=True, key=f"fmt_{df_key}")
                    
                    ai_tab1, ai_tab2 = st.tabs(["🤖 模式 1：内置 AI 一键直出", "📋 模式 2：复制 Prompt 给第三方 AI"])
                    
                    with ai_tab1:
                        st.info("💡 站长已为您内置专属 AI 算力，点击下方按钮即可一键编纂制卡数据！")
                        custom_prompt = st.text_area("📝 自定义 AI Prompt (可修改)", value=get_base_prompt_template(export_format), height=250, key=f"prompt_{df_key}")
                        
                        if st.button("⚡ 召唤 DeepSeek 立即生成卡片", key=f"btn_{df_key}", type="primary"):
                            with st.spinner("AI 正在云端光速编纂卡片，请稍候..."):
                                ai_result = call_deepseek_api(custom_prompt, pure_words)
                                
                                if "❌" in ai_result or "🚨" in ai_result or "⏳" in ai_result:
                                    st.error(ai_result)
                                else:
                                    st.success("🎉 生成完成！请务必通过下方按钮下载，直接导入 Anki。")
                                    
                                    # 极其关键的 utf-8-sig 编码修复，保证 Anki 导入绝不乱码
                                    mime_type = "text/csv" if export_format == "CSV" else "text/plain"
                                    st.download_button(
                                        label=f"📥 一键下载标准 Anki 导入文件 (.{export_format.lower()})", 
                                        data=ai_result.encode('utf-8-sig'), 
                                        file_name=f"anki_cards_{label}.{export_format.lower()}", 
                                        mime=mime_type,
                                        type="primary",
                                        use_container_width=True
                                    )
                                    
                                    st.markdown("##### 📝 预览框 (仅供查看，请勿从此处复制粘贴)")
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
        
        with t_raw:
            st.info("💡 这是自动词形还原后的全文输出，已针对长文优化防卡死体验。")
            st.download_button(label="💾 一键下载完整词形还原原文 (.txt)", data=st.session_state.lemma_text, file_name="lemmatized_text.txt", mime="text/plain", type="primary")
            if len(st.session_state.lemma_text) > 50000:
                st.warning("⚠️ 文本超长，仅展示前 50,000 字符。")
                st.code(st.session_state.lemma_text[:50000] + "\n\n... [请下载查看完整内容] ...", language='text')
            else:
                st.code(st.session_state.lemma_text, language='text')