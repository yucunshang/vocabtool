import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import time

# ==========================================
# 0. 基础配置 (移动端优化)
# ==========================================
st.set_page_config(
    page_title="Vocab Master", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* 界面紧凑 */
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    
    /* 按钮大尺寸 */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; font-size: 16px !important;
        margin-top: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* 提示框圆角 */
    .stAlert { border-radius: 10px; }
    
    /* 调整 Tab 样式 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 8px; padding: 0 10px; }
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
            
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.dropna(subset=[r_col])
            df = df.sort_values(r_col)
            
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            return vocab_dict, df, r_col, w_col
        except Exception as e:
            st.error(f"数据加载出错: {e}")
            return {}, None, None, None
    return {}, None, None, None

VOCAB_DICT, FULL_DF, RANK_COL, WORD_COL = load_data()
def get_lemma(word): return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. 核心算法 (极速优化版)
# ==========================================
def classify_words_fast(text, current_lvl, target_lvl):
    """
    极速处理逻辑：
    1. 正则提取
    2. 立即 Set 去重 (速度提升核心)
    3. 只对唯一单词进行词形还原和查询
    """
    # 1. 快速正则提取 + 小写
    raw_words = re.findall(r"[a-z]+", text.lower())
    total_count = len(raw_words)
    
    # 2. 立即去重 (例如文章 1万词，去重后可能只有 800 词)
    unique_words = set(raw_words)
    
    mastered, target, beyond = [], [], []
    
    # 3. 仅循环唯一单词
    for w in unique_words:
        if len(w) < 2: continue # 忽略单字母
        
        # 还原 (耗时操作，现在次数少了很多)
        lemma = get_lemma(w)
        
        # 查表
        rank = VOCAB_DICT.get(lemma, 99999)
        
        # 分类 (保留 rank 以便后续排序)
        if rank <= current_lvl:
            mastered.append((lemma, rank))
        elif current_lvl < rank <= target_lvl:
            target.append((lemma, rank))
        else:
            beyond.append((lemma, rank))
            
    # 4. 排序
    mastered = sorted(list(set(mastered)), key=lambda x: x[1])
    target = sorted(list(set(target)), key=lambda x: x[1])
    beyond = sorted(list(set(beyond)), key=lambda x: x[1])
    
    return total_count, mastered, target, beyond

def generate_prompt(words, settings):
    # Prompt 不包含 rank，只取单词
    clean_words = [w for w, r in words]
    word_list_str = ", ".join(clean_words)
    
    fmt = settings.get("format", "CSV")
    ex_count = settings.get("example_count", 1)
    lang = settings.get("lang", "Chinese")
    
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
   - Example Style: **Start with UPPERCASE**. Wrapped in <em>.
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

# ==========================================
# 3. 主界面
# ==========================================
st.title("⚡️ Vocab Master")

if FULL_DF is None:
    st.error("⚠️ 缺少词频文件 (coca_cleaned.csv)")
else:
    # --- 全局设置 ---
    with st.expander("⚙️ 生成设置 (Prompt Settings)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            set_format = st.selectbox("导出格式", ["CSV", "TXT"], index=0)
            set_lang = st.selectbox("释义语言", ["Chinese", "English"], index=0)
        with c2:
            set_ex_count = st.number_input("例句数量", 1, 3, 1)
            set_case = st.selectbox("风格", ["Front:Phrase (Lower)", "Front:Word"], index=0)
    
    settings = {"format": set_format, "lang": set_lang, "example_count": set_ex_count}

    # --- 功能导航 ---
    mode = st.radio("Mode", ["🔢 词频刷词", "📖 文本提取", "🛠️ 格式转换"], horizontal=True, label_visibility="collapsed")
    
    # ------------------------------------------------
    # 模式 1: 刷词
    # ------------------------------------------------
    if mode == "🔢 词频刷词":
        c1, c2 = st.columns(2)
        with c1: start_rank = st.number_input("起始排名", 8000, step=50)
        with c2: count = st.number_input("生成数量", 50, step=10)
        
        # 🟢 Rank 按钮
        show_rank = st.checkbox("在列表中显示排名 (Show Rank)", value=False)

        filtered = FULL_DF[FULL_DF[RANK_COL] >= start_rank].sort_values(RANK_COL).head(count)
        
        # 转换为 (word, rank) 元组列表
        word_data = list(zip(filtered[WORD_COL], filtered[RANK_COL]))
        
        if word_data:
            # 准备显示文本
            if show_rank:
                display_text = ", ".join([f"{w} ({int(r)})" for w, r in word_data])
            else:
                display_text = ", ".join([w for w, r in word_data])
                
            st.info(f"提取 {len(word_data)} 个单词 ({int(word_data[0][1])}-{int(word_data[-1][1])})")
            
            # 🟢 使用 st.code 实现一键复制
            st.code(display_text, language="text")
            st.caption("👆 点击右上角图标复制")
            
            if st.button("🚀 生成 Prompt", type="primary"):
                prompt = generate_prompt(word_data, settings)
                st.code(prompt, language="markdown")
        else:
            st.warning("无数据")

    # ------------------------------------------------
    # 模式 2: 提取 (极速版)
    # ------------------------------------------------
    elif mode == "📖 文本提取":
        st.caption("分析文章，极速分级")
        
        c1, c2 = st.columns(2)
        with c1: curr = st.number_input("当前水平", 4000, step=500)
        with c2: targ = st.number_input("目标水平", 8000, step=500)
        
        inp = st.radio("输入", ["粘贴", "上传"], horizontal=True, label_visibility="collapsed")
        text = ""
        if inp == "粘贴": text = st.text_area("文本", height=150)
        else: 
            up = st.file_uploader("文件 (TXT/PDF/DOCX)", type=["txt","pdf","docx"])
            if up: text = extract_text(up)
        
        # 🟢 Rank 按钮
        show_rank_ext = st.checkbox("列表显示排名 (Show Rank)", value=False)

        if text and st.button("🔍 开始分析", type="primary"):
            # 🟢 进度反馈 + 计时
            with st.spinner("正在极速分析中..."):
                t0 = time.time()
                # 调用优化后的函数
                total_words, m_list, t_list, b_list = classify_words_fast(text, curr, targ)
                t1 = time.time()
            
            st.success(f"✅ 分析完成！处理 {total_words} 词，耗时 {t1-t0:.3f} 秒")
            
            # 准备显示函数
            def get_display_str(data_list):
                if show_rank_ext:
                    return ", ".join([f"{w} ({int(r)})" for w, r in data_list])
                else:
                    return ", ".join([w for w, r in data_list])

            # Tabs
            t1, t2, t3 = st.tabs([f"🎯 重点 ({len(t_list)})", f"✅ 已掌握 ({len(m_list)})", f"🚀 超纲 ({len(b_list)})"])
            
            # --- 重点 ---
            with t1:
                if t_list:
                    st.markdown("##### 🎯 重点背诵")
                    # 🟢 st.code 实现复制
                    st.code(get_display_str(t_list), language="text")
                    
                    # 🟢 数量预警
                    if len(t_list) > 200:
                        st.error(f"⚠️ 单词数量 ({len(t_list)}) 较多！AI 可能无法一次性生成所有卡片。建议分多次复制。")
                    
                    if st.button("🚀 生成 Prompt (重点词)"):
                        prompt = generate_prompt(t_list, settings)
                        st.code(prompt, language="markdown")
                else: st.info("无")
            
            # --- 已掌握 ---
            with t2:
                if m_list:
                    with st.expander("查看列表"):
                        st.code(get_display_str(m_list), language="text")
                else: st.write("无")
                
            # --- 超纲 ---
            with t3:
                if b_list:
                    with st.expander("查看列表"):
                        st.code(get_display_str(b_list), language="text")
                else: st.write("无")

    # ------------------------------------------------
    # 模式 3: 转换
    # ------------------------------------------------
    elif mode == "🛠️ 格式转换":
        st.markdown("### 📥 转 Anki CSV")
        st.caption("粘贴 AI 回复 (无表头)，自动转文件")
        
        csv_in = st.text_area("粘贴", height=200)
        
        if csv_in:
            st.download_button(
                "📥 下载 .csv",
                csv_in.strip().encode('utf-8'),
                "anki_import.csv",
                "text/csv",
                type="primary"
            )