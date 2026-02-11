import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import io

# ==========================================
# 0. 基础配置与 CSS (适配手机)
# ==========================================
st.set_page_config(
    page_title="Vocab Master", 
    page_icon="📱", 
    layout="centered",  # 手机端改为 centered 更聚焦
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* 界面紧凑化 */
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    
    /* 按钮优化：大尺寸适合手指点击 */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; font-size: 16px !important;
        margin-top: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* 文本框优化：方便复制，防止字体过小 */
    .stTextArea textarea { font-size: 14px !important; border-radius: 10px; }
    
    /* 设置栏样式 */
    [data-testid="stExpander"] { border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px; }
    
    /* 标签页优化 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载
# ==========================================
@st.cache_resource
def setup_nltk():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    try: 
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    except: pass
setup_nltk()

@st.cache_data
def load_data():
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            w_col = next((c for c in cols if 'word' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c), cols[1])
            
            # 清洗与类型转换
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.dropna(subset=[r_col])
            df = df.sort_values(r_col)
            
            # 字典：用于快速查询 Rank
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            return vocab_dict, df, r_col, w_col
        except Exception as e:
            st.error(f"数据加载出错: {e}")
            return {}, None, None, None
    return {}, None, None, None

VOCAB_DICT, FULL_DF, RANK_COL, WORD_COL = load_data()
def get_lemma(word): return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. 动态 Prompt 生成器 (基于全局设置)
# ==========================================
def generate_dynamic_prompt(words, settings):
    word_list_str = ", ".join(words)
    
    # 解析设置
    fmt = settings.get("format", "CSV")
    ex_count = settings.get("example_count", 1)
    lang = settings.get("lang", "Chinese")
    
    # 这里的 Prompt 严格遵循您之前的要求：
    # 1. 小写短语
    # 2. 词源只解释组件
    # 3. 例句大写
    # 4. 无表头
    
    prompt = f"""Role: High-Efficiency Anki Card Creator
Task: Convert the provided word list into a strict {fmt} data block.

--- OUTPUT FORMAT RULES ---
1. Structure: {'2 Columns (Front, Back)' if fmt=='CSV' else 'Custom Text Format'}.
   Format: "Front","Back"
   Header: **Do NOT output a header row.**

2. Column 1 (Front):
   - Content: A natural, short English phrase/collocation.
   - Style: **ALL LOWERCASE**.

3. Column 2 (Back):
   - Content: Definition + {ex_count} Example(s) + Etymology.
   - HTML Layout: Definition <br> <br> <em>Example</em> <br> <br> 【源】Etymology
   - Definition Language: {lang} & English concise (Start with lowercase).
   - Example Style: **Start with UPPERCASE** (Normal sentence case). Wrapped in <em>.
   - Spacing: Double <br> tags.

4. Etymology Style:
   - Only explain roots/affixes in {lang}.
   - Format: 【源】Root (Meaning) + Affix (Meaning)
   - Do NOT explain the final word meaning.

5. Atomicity: Separate rows for distinct meanings.

--- WORD LIST ---
{word_list_str}
"""
    return prompt

# ==========================================
# 3. 辅助功能
# ==========================================
def extract_text(file_obj):
    try:
        ext = file_obj.name.split('.')[-1].lower()
        if ext == 'txt': return file_obj.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            import PyPDF2; reader = PyPDF2.PdfReader(file_obj)
            return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == 'docx':
            import docx; doc = docx.Document(file_obj)
            return " ".join([p.text for p in doc.paragraphs])
    except: return ""
    return ""

def classify_words(text, current_lvl, target_lvl):
    raw_words = re.findall(r"[a-zA-Z']+", text)
    lemmas = set(get_lemma(w).lower() for w in raw_words if len(w)>=2)
    
    mastered, target, beyond = [], [], []
    
    for w in lemmas:
        rank = VOCAB_DICT.get(w, 99999) # 99999 = Unknown/Rare
        
        if rank <= current_lvl:
            mastered.append((w, rank))
        elif current_lvl < rank <= target_lvl:
            target.append((w, rank))
        else:
            beyond.append((w, rank))
            
    # 排序
    mastered.sort(key=lambda x: x[1])
    target.sort(key=lambda x: x[1])
    beyond.sort(key=lambda x: x[1])
    
    return [x[0] for x in mastered], [x[0] for x in target], [x[0] for x in beyond]

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.title("⚡️ Vocab Master")

if FULL_DF is None:
    st.error("⚠️ 缺少词频文件 (coca_cleaned.csv)")
else:
    # ------------------------------------------------
    # 🟢 需求3：折叠栏放在所有功能上面 (全局设置)
    # ------------------------------------------------
    with st.expander("⚙️ 生成设置 (Prompt Settings)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            set_format = st.selectbox("导出格式", ["CSV", "TXT"], index=0)
            set_lang = st.selectbox("释义语言", ["Chinese", "English"], index=0)
        with c2:
            set_ex_count = st.number_input("例句数量", 1, 3, 1)
            set_case = st.selectbox("风格", ["Front:Phrase (Lower)", "Front:Word"], index=0)
    
    # 包装设置
    global_settings = {
        "format": set_format,
        "lang": set_lang,
        "example_count": set_ex_count
    }

    # --- 导航 ---
    mode = st.radio("功能模式", ["🔢 词频刷词", "📖 文本提取", "🛠️ 格式转换"], horizontal=True, label_visibility="collapsed")
    
    # ------------------------------------------------
    # 模式 1: 刷词 (Range)
    # ------------------------------------------------
    if mode == "🔢 词频刷词":
        st.caption("按排名批量生成单词卡")
        c1, c2 = st.columns(2)
        with c1: start_rank = st.number_input("起始排名", 8000, step=50)
        with c2: count = st.number_input("生成数量", 50, step=10)
            
        filtered = FULL_DF[FULL_DF[RANK_COL] >= start_rank].sort_values(RANK_COL).head(count)
        words = filtered[WORD_COL].tolist()
        
        if words:
            real_range = f"{int(filtered.iloc[0][RANK_COL])}-{int(filtered.iloc[-1][RANK_COL])}"
            st.info(f"提取 {len(words)} 个单词 ({real_range})")
            
            # 🟢 需求1：单词列表要可以复制
            st.text_area("📋 单词列表 (全选复制)", ", ".join(words), height=100)
            
            if st.button("🚀 生成 Prompt", type="primary"):
                prompt = generate_dynamic_prompt(words, global_settings)
                st.code(prompt, language="markdown")
                st.success("点击右上角复制 -> 发给 AI")
        else:
            st.warning("无数据")

    # ------------------------------------------------
    # 模式 2: 提取 (Extract) - 🟢 需求2：三栏功能回归
    # ------------------------------------------------
    elif mode == "📖 文本提取":
        st.caption("分析文章，按你的词汇量分级 (无 Top N 限制)")
        
        # 词汇量设置
        col_a, col_b = st.columns(2)
        with col_a: 
            curr_lvl = st.number_input("当前词汇量 (Current)", value=4000, step=500)
        with col_b: 
            targ_lvl = st.number_input("目标词汇量 (Target)", value=8000, step=500)
        
        # 输入
        inp_type = st.radio("Input", ["粘贴文本", "上传文件"], horizontal=True, label_visibility="collapsed")
        raw_text = ""
        if inp_type == "粘贴文本":
            raw_text = st.text_area("在此粘贴", height=150)
        else:
            up = st.file_uploader("支持 TXT/PDF/DOCX", type=["txt","pdf","docx"])
            if up: raw_text = extract_text(up)
            
        # 处理与展示
        if raw_text and st.button("🔍 分析单词", type="primary"):
            w_mastered, w_target, w_beyond = classify_words(raw_text, curr_lvl, targ_lvl)
            
            # 🟢 需求2：三栏展示 (手机端用 Tabs 最合适)
            tab1, tab2, tab3 = st.tabs([
                f"🎯 重点 ({len(w_target)})", 
                f"✅ 已掌握 ({len(w_mastered)})", 
                f"🚀 超纲 ({len(w_beyond)})"
            ])
            
            # --- Tab 1: 重点单词 ---
            with tab1:
                if w_target:
                    st.success("这些是你当前阶段最需要背的词！")
                    # 🟢 需求2：可以复制，折叠
                    with st.expander("📋 展开/复制列表", expanded=True):
                        st.text_area("Target Words", ", ".join(w_target), height=150, key="txt_target")
                    
                    if st.button("🚀 为重点词生成 Prompt"):
                        prompt = generate_dynamic_prompt(w_target, global_settings)
                        st.code(prompt, language="markdown")
                else:
                    st.info("此区间无单词")

            # --- Tab 2: 已掌握 ---
            with tab2:
                if w_mastered:
                    st.caption("低于当前词汇量的词：")
                    with st.expander("📋 展开/复制列表"):
                        st.text_area("Mastered Words", ", ".join(w_mastered), height=150, key="txt_mastered")
                else: st.write("无")

            # --- Tab 3: 超纲 ---
            with tab3:
                if w_beyond:
                    st.caption("高于目标词汇量或生僻词：")
                    with st.expander("📋 展开/复制列表"):
                        st.text_area("Beyond Words", ", ".join(w_beyond), height=150, key="txt_beyond")
                else: st.write("无")

    # ------------------------------------------------
    # 模式 3: 转换
    # ------------------------------------------------
    elif mode == "🛠️ 格式转换":
        st.markdown("### 📥 AI 结果转 Anki CSV")
        st.caption("粘贴 AI 返回的纯数据 (No Header)，自动下载")
        
        csv_in = st.text_area("粘贴内容", height=200, placeholder='"phrase","def..."')
        
        if csv_in:
            csv_str = csv_in.strip()
            # 保持纯净，不加 Header
            st.download_button(
                "📥 下载 .csv (纯数据)",
                csv_str.encode('utf-8'),
                "anki_import.csv",
                "text/csv",
                type="primary"
            )