import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import io

# ==========================================
# 0. 基础配置
# ==========================================
st.set_page_config(
    page_title="Prompt Gen", 
    page_icon="📱", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; font-size: 18px !important;
        margin-top: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stTextArea>div>div>textarea { font-size: 16px !important; border-radius: 10px; }
    .stNumberInput input { font-size: 18px !important; }
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
    """
    加载词频数据，保留所有行（不去重），以解决一词多义导致的漏词问题。
    """
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            
            w_col = next((c for c in cols if 'word' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c), cols[1])
            
            # 清洗
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.dropna(subset=[r_col])
            
            # 排序
            df = df.sort_values(r_col)
            
            # 1. 字典：Word -> Rank (供文本提取用)
            # 这里如果遇到重复词，默认保留最后的（或者任意一个，影响不大）
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            
            # 2. DataFrame (供刷词用，保留所有行)
            return vocab_dict, df, r_col, w_col
            
        except Exception as e:
            st.error(f"数据加载出错: {e}")
            return {}, None, None, None
    return {}, None, None, None

VOCAB_DICT, FULL_DF, RANK_COL, WORD_COL = load_data()

def get_lemma(word): return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. Prompt 生成逻辑 (更新：小写 + 详细词源)
# ==========================================
def generate_strict_prompt(words):
    word_list_str = ", ".join(words)
    prompt = f"""Role: High-Efficiency Anki Card Creator
Task: Convert the provided word list into a strict CSV data block.

--- OUTPUT FORMAT RULES ---
1. Structure: 2 Columns only. Comma-separated. All fields double-quoted.
   Format: "Front","Back"
   Header: **Do NOT output a header row.** Only output the data rows.

2. Column 1 (Front):
   - Content: A natural, short English phrase or collocation containing the target word.
   - Style: **ALL LOWERCASE** (do not capitalize the first letter). 
   - Example: "a limestone quarry", not "A limestone quarry".

3. Column 2 (Back):
   - Content: Definition + Example + Etymology.
   - HTML Layout: Definition <br> <br> <em>Example Sentence</em> <br> <br> 【源】Etymology
   - definition style: Concise English, **start with lowercase**.
   - example style: Wrapped in <em>, **start with lowercase**.
   - Spacing: Use double <br> tags ( <br> <br> ) between sections.

4. Etymology Style (Detailed):
   - Format: 【源】Root (Chinese Meaning) + Affix (Chinese Meaning) → Logic.
   - Requirement: **MUST provide the Chinese meaning** for roots/affixes.
   - Example 1: 【源】pro- (向前) + gress (走) → 前进
   - Example 2: 【源】Lat. 'vigere' (活跃) → 精力

5. Atomicity Principle (Strict):
   - If a word has distinct meanings, **generate SEPARATE rows**.

6. Output: 
   - Code Block ONLY. 
   - NO header line.

--- WORD LIST ---
{word_list_str}
"""
    return prompt

# ==========================================
# 3. 辅助功能
# ==========================================
def extract_text_from_file(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'txt': return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            import PyPDF2; reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == 'docx':
            import docx; doc = docx.Document(uploaded_file)
            return " ".join([p.text for p in doc.paragraphs])
    except: return ""
    return ""

def process_text_input(text, min_rank, max_rank):
    words = re.findall(r"[a-zA-Z']+", text)
    lemmas = set(get_lemma(w).lower() for w in words if len(w)>=2)
    filtered = [(w, VOCAB_DICT.get(w, 99999)) for w in lemmas]
    filtered = [w for w, r in filtered if min_rank <= r <= max_rank]
    filtered.sort(key=lambda w: VOCAB_DICT.get(w, 99999))
    return filtered

# ==========================================
# 4. 主界面
# ==========================================
st.title("⚡️ Anki Master")

if FULL_DF is None:
    st.error("⚠️ 缺少词频文件 (coca_cleaned.csv)")
else:
    mode = st.radio("功能", ["🔢 刷词", "📖 提取", "🛠️ 转换"], horizontal=True, label_visibility="collapsed")
    
    # ------------------------------------------------
    # 模式 1: 刷词 (凑单模式)
    # ------------------------------------------------
    if mode == "🔢 刷词":
        st.caption("从指定排名开始，自动凑齐数量")
        
        col1, col2 = st.columns(2)
        with col1:
            start_rank = st.number_input("起始排名", value=8000, step=50)
        with col2:
            count = st.number_input("生成数量", value=50, step=10)
            
        # 逻辑：筛选 >= start_rank 的所有词，排序，取前 count 个
        filtered_df = FULL_DF[FULL_DF[RANK_COL] >= start_rank].sort_values(RANK_COL)
        selected_df = filtered_df.head(count)
        target_words = selected_df[WORD_COL].tolist()
        
        if target_words:
            real_start = int(selected_df.iloc[0][RANK_COL])
            real_end = int(selected_df.iloc[-1][RANK_COL])
            
            st.info(f"✅ 已提取 **{len(target_words)}** 个单词")
            st.caption(f"实际排名范围: {real_start} - {real_end}")
            
            with st.expander("👀 查看单词列表"):
                st.text(", ".join(target_words))

            if st.button("🚀 生成 Prompt"):
                prompt = generate_strict_prompt(target_words)
                st.code(prompt, language="markdown")
                st.success("请复制上方代码 -> 发送给 ChatGPT")
        else:
            st.warning("该排名之后没有更多单词了。")

    # ------------------------------------------------
    # 模式 2: 提取
    # ------------------------------------------------
    elif mode == "📖 提取":
        inp = st.radio("方式", ["粘贴", "上传"], horizontal=True, label_visibility="collapsed")
        txt = ""
        target_words = []
        
        if inp == "粘贴": txt = st.text_area("文本", height=100)
        else: 
            up = st.file_uploader("文件", type=["txt","pdf","docx"])
            if up: txt = extract_text_from_file(up)
        
        if txt and st.button("提取"):
            target_words = process_text_input(txt, 3000, 20000)
            st.session_state['temp_ext'] = target_words
        
        if 'temp_ext' in st.session_state: target_words = st.session_state['temp_ext']

        if target_words:
            if len(target_words)>100: 
                target_words=target_words[:100]
                st.warning("已截取前 100 个")
            
            st.info(f"提取到 {len(target_words)} 个生词")
            if st.button("🚀 生成 Prompt"):
                prompt = generate_strict_prompt(target_words)
                st.code(prompt, language="markdown")

    # ------------------------------------------------
    # 模式 3: 转换
    # ------------------------------------------------
    elif mode == "🛠️ 转换":
        st.markdown("### 📥 AI 结果转 Anki 文件")
        st.caption("自动补全表头，支持 Anki 直接导入")
        
        csv_input = st.text_area("粘贴内容", height=200, placeholder='"phrase","def..."')
        
        if csv_input:
            csv_content = csv_input.strip()
            # 自动补全 Header
            if '"Front","Back"' not in csv_content and "Front,Back" not in csv_content:
                final_csv = '"Front","Back"\n' + csv_content
            else:
                final_csv = csv_content
            
            st.download_button(
                label="📥 下载 .csv (Anki Ready)",
                data=final_csv.encode('utf-8'),
                file_name="anki_import.csv",
                mime="text/csv",
                type="primary"
            )
            st.success("下载后 -> 分享到 Anki -> 直接 Import")