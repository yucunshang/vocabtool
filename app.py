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

# 移动端 CSS 优化
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
    vocab_dict = {}; rank_map = {}
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
            df = df.dropna(subset=[r_col]).sort_values(r_col).drop_duplicates(subset=[w_col])
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            for w, r in vocab_dict.items():
                if int(r) not in rank_map: rank_map[int(r)] = []
                rank_map[int(r)].append(w)
        except: pass
    return vocab_dict, rank_map

VOCAB_DICT, RANK_MAP = load_data()
def get_lemma(word): return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. Prompt 生成逻辑
# ==========================================
def generate_strict_prompt(words):
    word_list_str = ", ".join(words)
    prompt = f"""Role: High-Efficiency Anki Card Creator
Task: Convert the provided word list into a strict CSV code block.

--- OUTPUT FORMAT RULES ---
1. Structure: 2 Columns only. Comma-separated. All fields double-quoted.
   Format: "Front","Back"
   Header: MUST include a header row: "Front","Back"

2. Column 1 (Front):
   - Content: A natural, short English phrase or collocation containing the target word.
   - Style: Plain text.

3. Column 2 (Back):
   - Content: Definition + Example + Etymology.
   - HTML Layout: Definition<br><em>Example Sentence</em><br>【源】Etymology
   - Constraints: Use <br> for breaks. Wrap example in <em>.

4. Atomicity: Separate rows for distinct meanings.
5. Output: Code Block ONLY. Start with the header.

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

if not VOCAB_DICT:
    st.error("⚠️ 缺少词频文件 (coca_cleaned.csv)")
else:
    # 增加了一个 "🛠️ 转换" 标签
    mode = st.radio("功能", ["🔢 刷词", "📖 提取", "🛠️ 转换"], horizontal=True, label_visibility="collapsed")
    
    # --- 模式 1 & 2: 生成 Prompt ---
    if mode in ["🔢 刷词", "📖 提取"]:
        target_words = []
        if mode == "🔢 刷词":
            c1, c2 = st.columns(2)
            with c1: s_r = st.number_input("Start", 8000, step=50)
            with c2: e_r = st.number_input("End", 8050, step=50)
            for r in range(s_r, e_r + 1):
                if r in RANK_MAP: target_words.extend(RANK_MAP[r])
            target_words = list(dict.fromkeys(target_words))
            if target_words: st.info(f"选中 {len(target_words)} 个单词")
            
        else: # 提取模式
            inp = st.radio("方式", ["粘贴", "上传"], horizontal=True, label_visibility="collapsed")
            txt = ""
            if inp == "粘贴": txt = st.text_area("文本", height=100)
            else: 
                up = st.file_uploader("文件", type=["txt","pdf","docx"])
                if up: txt = extract_text_from_file(up)
            
            if txt and st.button("提取"):
                target_words = process_text_input(txt, 3000, 20000)
                st.session_state['temp'] = target_words
            
            if 'temp' in st.session_state: target_words = st.session_state['temp']

        if target_words:
            if len(target_words)>100: 
                target_words=target_words[:100]
                st.warning("已截取前 100 个")
                
            if st.button("🚀 生成 Prompt"):
                prompt = generate_strict_prompt(target_words)
                st.code(prompt, language="markdown")
                st.success("复制上方代码 -> 发给 ChatGPT -> 复制 ChatGPT 的结果回来")

    # --- 模式 3: 格式转换 (AI -> Anki File) ---
    elif mode == "🛠️ 转换":
        st.markdown("### 📥 AI 结果转 Anki 文件")
        st.caption("解决手机无法保存 CSV 的问题。步骤：\n1. 复制 ChatGPT 生成的代码块内容\n2. 粘贴到下方\n3. 下载文件并在 Anki 打开")
        
        csv_input = st.text_area("在此粘贴 ChatGPT 生成的 CSV 内容", height=200, placeholder='"Front","Back"\n"phrase 1","def 1..."')
        
        if csv_input:
            # 简单清洗，防止首尾空行
            csv_content = csv_input.strip()
            
            # 检查是否有 header，如果没有强行加一个，如果有保留
            # 简单的检查方法：看第一行是否包含 "Front"
            if "front" not in csv_content.split('\n')[0].lower():
                csv_content = '"Front","Back"\n' + csv_content
            
            # 转换为字节流
            csv_bytes = csv_content.encode('utf-8')
            
            st.download_button(
                label="📥 下载 .csv (直接导入 Anki)",
                data=csv_bytes,
                file_name="anki_import.csv",
                mime="text/csv",
                type="primary"
            )
            
            st.markdown("""
            **iOS 导入教程：**
            1. 点击上方按钮下载。
            2. 浏览器弹出“下载”，点击下载。
            3. 点击浏览器地址栏左侧的 **"大小" (Aa)** -> **下载项**。
            4. 点击 `anki_import.csv`。
            5. 点击右上角 **分享图标** -> 选择 **Anki** 图标。
            6. Anki 会自动打开，直接点 **Import** 即可（无需设置，因为我们已经加了表头）。
            """)