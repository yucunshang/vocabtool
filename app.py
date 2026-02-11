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
    修复版加载逻辑：
    1. 不再对单词进行去重 (drop_duplicates)。
    2. 允许同一个单词出现在不同的排名（解决一词多义导致的漏词问题）。
    """
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    vocab_dict = {} # Word -> Rank (供文本提取模式用，默认保留第一次出现的Rank)
    rank_map = {}   # Rank -> [Words] (供刷词模式用，保留所有)

    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            
            w_col = next((c for c in cols if 'word' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c), cols[1])
            
            # 基础清洗：去空、小写
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.dropna(subset=[r_col])
            
            # 【关键修改】不再执行 drop_duplicates(subset=[w_col])
            # 我们保留所有行，确保 Rank 8000 的 splash 也能被索引到
            
            # 构建 Rank -> Word 列表 (一对多)
            for index, row in df.iterrows():
                r = int(row[r_col])
                w = row[w_col]
                if r not in rank_map:
                    rank_map[r] = []
                rank_map[r].append(w)
            
            # 构建 Word -> Rank (文本模式用)
            # 这里如果为了严谨，我们倒序遍历，保留排名靠前的那个
            # 或者直接由 pandas 默认处理
            df_unique = df.sort_values(r_col).drop_duplicates(subset=[w_col])
            vocab_dict = pd.Series(df_unique[r_col].values, index=df_unique[w_col]).to_dict()

        except Exception as e:
            st.error(f"数据加载出错: {e}")
            
    return vocab_dict, rank_map

VOCAB_DICT, RANK_MAP = load_data()

def get_lemma(word): return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. Prompt 生成逻辑
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
   - Style: Plain text.

3. Column 2 (Back):
   - Content: Definition + Example + Etymology.
   - HTML Layout: Definition <br> <br> <em>Example Sentence</em> <br> <br> 【源】Etymology
   - Constraints: 
     - Use double <br> tags ( <br> <br> ) between sections to ensure clear visual spacing.
     - Example sentence must be wrapped in <em> tags.

4. Atomicity Principle (Strict):
   - If a word has distinct meanings, **generate SEPARATE rows**.

5. Output: 
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

if not RANK_MAP:
    st.error("⚠️ 缺少词频文件或加载失败")
else:
    mode = st.radio("功能", ["🔢 刷词", "📖 提取", "🛠️ 转换"], horizontal=True, label_visibility="collapsed")
    
    # ------------------------------------------------
    # 模式 1: 刷词 (纯净版)
    # ------------------------------------------------
    if mode == "🔢 刷词":
        st.caption("从指定排名提取 (保留重复词)")
        
        col1, col2 = st.columns(2)
        with col1:
            start_rank = st.number_input("起始排名", value=8000, step=50)
        with col2:
            end_rank = st.number_input("结束排名", value=8050, step=50)
            
        if start_rank >= end_rank:
            st.warning("范围错误")
        else:
            target_words = []
            # 简单遍历范围，直接取 map 里的值
            for r in range(start_rank, end_rank + 1):
                if r in RANK_MAP:
                    target_words.extend(RANK_MAP[r])
            
            # 不再进行去重 (dict.fromkeys)，保留所有提取到的词
            
            if target_words:
                st.info(f"✅ 区间 {start_rank}-{end_rank} 提取到 **{len(target_words)}** 个单词")
                
                # 预览
                with st.expander("查看单词列表"):
                    st.text(", ".join(target_words))

                if st.button("🚀 生成 Prompt"):
                    prompt = generate_strict_prompt(target_words)
                    st.code(prompt, language="markdown")
                    st.success("请复制上方代码 -> 发送给 ChatGPT")
            else:
                st.warning("该区间没有单词。")

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
        st.caption("粘贴 ChatGPT 的代码块，自动生成标准 CSV")
        
        csv_input = st.text_area("粘贴内容", height=200, placeholder='"phrase","def..."')
        
        if csv_input:
            csv_content = csv_input.strip()
            # 自动补全 Header
            if '"Front","Back"' not in csv_content and "Front,Back" not in csv_content:
                final_csv = '"Front","Back"\n' + csv_content
            else:
                final_csv = csv_content
            
            st.download_button(
                label="📥 下载 .csv (自动补全格式)",
                data=final_csv.encode('utf-8'),
                file_name="anki_import.csv",
                mime="text/csv",
                type="primary"
            )
            st.success("下载后 -> 分享到 Anki -> 直接 Import")