import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import time

# ==========================================
# 0. 基础配置与移动端适配
# ==========================================
st.set_page_config(
    page_title="Prompt Gen", 
    page_icon="📱", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 移动端 CSS 深度优化
st.markdown("""
<style>
    /* 1. 全局容器：减少留白，适应手机屏 */
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 3rem; 
        padding-left: 1rem; 
        padding-right: 1rem;
    }
    
    /* 2. 隐藏无关元素 (顶部条、页脚、侧边栏按钮) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    
    /* 3. 按钮：大尺寸，圆角，适合手指点击 */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        font-weight: bold;
        font-size: 18px !important;
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* 4. 输入框：防止 iOS 自动缩放 (字体需>=16px) */
    .stTextArea>div>div>textarea {
        font-size: 16px !important; 
        border-radius: 10px;
    }
    .stNumberInput input {
        font-size: 18px !important;
    }
    
    /* 5. 提示框美化 */
    .stAlert {
        border-radius: 10px;
    }
    
    /* 6. 代码块：紧凑模式 */
    .stCode {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载与 NLTK 初始化
# ==========================================
@st.cache_resource
def setup_nltk():
    """下载必要的 NLTK 数据包"""
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
    """加载词频表和过滤表"""
    vocab_dict = {}     # Word -> Rank
    rank_map = {}       # Rank -> List of Words
    
    # 1. 尝试加载词频表 (支持多种文件名)
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # 自动识别列名
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            
            # 寻找 word 和 rank 列
            w_col = next((c for c in cols if 'word' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c), cols[1])
            
            # 清洗
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            # 确保 Rank 是数字
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.dropna(subset=[r_col])
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col])
            
            # 构建字典
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            
            # 构建反向映射 (Rank -> Word List)
            for w, r in vocab_dict.items():
                r = int(r)
                if r not in rank_map: rank_map[r] = []
                rank_map[r].append(w)
                
        except Exception as e:
            st.error(f"数据加载失败: {e}")
    
    return vocab_dict, rank_map

VOCAB_DICT, RANK_MAP = load_data()

def get_lemma(word):
    """获取单词原形"""
    return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. 核心功能：Prompt 生成器 (最终优化版)
# ==========================================
def generate_strict_prompt(words):
    word_list_str = ", ".join(words)
    count = len(words)
    
    # 这是您要求的、不考虑 Token 限制、追求最高质量的 Prompt
    prompt = f"""Role: High-Efficiency Anki Card Creator
Task: Convert the provided word list into a strict CSV code block for Anki import.

--- OUTPUT FORMAT RULES ---

1. Structure: 2 Columns only. Comma-separated. All fields double-quoted.
   Format: "Front","Back"

2. Column 1 (Front):
   - Content: A **natural, high-frequency English collocation or short phrase** containing the target word.
   - Goal: Maximize context retention.
   - Style: Plain text (NO bolding, NO extra symbols).

3. Column 2 (Back):
   - Content: Definition + Example + Etymology.
   - HTML Layout: Definition<br><em>Example Sentence</em><br>【源】Etymology
   - Constraints:
     - Use <br> tags for clear visual separation.
     - Example sentence must be wrapped in <em> tags (Italics) and be **natural/native-sounding**.
     - Definition: Precise and clear English definition matching the context of the phrase.

4. Etymology Style (Strict):
   - Language: CHINESE (中文).
   - Style: Logic-based, concise. Use arrows (→) to show the evolution of meaning.
   - Format: 【源】Root/Origin (Meaning) → Result/Logic.
   - Example: 【源】Lat. 'vigere' (活跃) → 精力/活力

5. Atomicity Principle (Crucial):
   - If a word has distinct meanings (e.g., Noun vs. Verb, or Literal vs. Metaphorical), **generate SEPARATE rows** for each distinct meaning. Do not combine them into one card.

6. Output Requirement:
   - Output the Code Block ONLY. No conversational text before or after.
   - Ensure specific CSV escaping if the content itself contains double quotes.

--- EXAMPLE OUTPUT ---
"limestone quarry","deep pit for extracting stone<br><em>The company owns a granite quarry.</em><br>【源】古法语 quarriere (方石) → 切石场"
"hunter's quarry","animal pursued by a hunter<br><em>The eagle spotted its quarry.</em><br>【源】古法语 cuir (皮革) → 放皮上的内脏赏赐 → 猎物"
"stiffen with cold","make or become rigid<br><em>His muscles began to stiffen.</em><br>【源】stiff (僵硬) + -en (使动)"

--- MY WORD LIST ({count} words) ---
{word_list_str}
"""
    return prompt

# ==========================================
# 3. 辅助功能：文本提取
# ==========================================
def extract_text_from_file(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'txt':
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            import docx
            doc = docx.Document(uploaded_file)
            return " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.error(f"文件读取错误: {str(e)}")
        return ""
    return ""

def process_text_input(text, min_rank, max_rank):
    # 1. 清洗
    words = re.findall(r"[a-zA-Z']+", text)
    # 2. 还原
    lemmas = set()
    for w in words:
        if len(w) < 2: continue
        lemma = get_lemma(w).lower()
        lemmas.add(lemma)
    
    # 3. 过滤 (根据 Rank)
    filtered = []
    for w in lemmas:
        rank = VOCAB_DICT.get(w, 99999) # 没找到的词视为生僻词(99999)
        if min_rank <= rank <= max_rank:
            filtered.append((w, rank))
            
    # 4. 排序 (按 Rank 排序，确保简单的在前或难的在前，这里默认按 Rank 升序)
    filtered.sort(key=lambda x: x[1])
    return [x[0] for x in filtered]

# ==========================================
# 4. 主界面逻辑 (APP UI)
# ==========================================

st.title("⚡️ Anki Prompt Gen")

# 检查数据是否加载
if not VOCAB_DICT:
    st.error("⚠️ 未找到词频数据 (coca_cleaned.csv)。请确保文件在根目录下。")
else:
    # 顶部导航 (类似 App Tab)
    mode = st.radio("模式", ["🔢 词频刷词", "📖 文本提取"], horizontal=True, label_visibility="collapsed")
    
    target_words = []
    
    # ----------------------
    # 模式 A: 词频范围生成
    # ----------------------
    if mode == "🔢 词频刷词":
        st.caption("适合每日定量刷词")
        col1, col2 = st.columns(2)
        with col1:
            start_r = st.number_input("Start Rank", value=8000, step=50)
        with col2:
            end_r = st.number_input("End Rank", value=8050, step=50)
            
        if start_r >= end_r:
            st.warning("开始排名必须小于结束排名")
        else:
            # 提取区间单词
            found_words = []
            for r in range(start_r, end_r + 1):
                if r in RANK_MAP:
                    found_words.extend(RANK_MAP[r])
            
            # 去重并保持顺序
            found_words = list(dict.fromkeys(found_words))
            
            if len(found_words) > 0:
                st.info(f"🎯 区间 {start_r}-{end_r} 命中 **{len(found_words)}** 个单词")
                
                # 预览
                with st.expander(f"预览列表 ({found_words[0]}...)", expanded=False):
                    st.text(", ".join(found_words))
                    
                target_words = found_words
            else:
                st.warning("该区间没有找到单词。")

    # ----------------------
    # 模式 B: 文本提取生成
    # ----------------------
    else:
        st.caption("从文章/字幕提取生词")
        
        input_method = st.radio("Input", ["粘贴", "上传"], horizontal=True, label_visibility="collapsed")
        
        raw_text = ""
        if input_method == "粘贴":
            raw_text = st.text_area("在此粘贴", height=150, placeholder="粘贴英文文章...")
        else:
            uploaded = st.file_uploader("文件 (TXT/PDF/DOCX)", type=["txt", "pdf", "docx"])
            if uploaded:
                raw_text = extract_text_from_file(uploaded)
                if raw_text: st.success("✅ 文件已读取")

        # 过滤设置
        with st.expander("⚙️ 难度过滤 (Rank)"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                min_filter = st.number_input("忽略简单词 (Top X)", value=3000, step=500)
            with f_col2:
                max_filter = st.number_input("忽略生僻词 (Bottom X)", value=20000, step=1000)
        
        if raw_text:
            if st.button("🔍 分析并提取", type="primary"):
                target_words = process_text_input(raw_text, min_filter, max_filter)
                if not target_words:
                    st.warning("未提取到符合条件的生词。")
                else:
                    st.success(f"筛选出 {len(target_words)} 个生词")
                    with st.expander("查看结果"):
                        st.text(", ".join(target_words))
                        # 临时保存到 session_state 这样不会刷新消失
                        st.session_state['temp_words'] = target_words

        if 'temp_words' in st.session_state and mode == "📖 文本提取":
             target_words = st.session_state['temp_words']

    # ----------------------
    # 结果生成区 (通用)
    # ----------------------
    if target_words:
        st.divider()
        
        # 批量处理建议
        MAX_BATCH = 100
        if len(target_words) > MAX_BATCH:
            st.warning(f"⚠️ 单词较多 ({len(target_words)}个)，建议分批。已自动截取前 {MAX_BATCH} 个。")
            target_words = target_words[:MAX_BATCH]
        
        if st.button("🚀 生成 Prompt (准备复制)", type="primary"):
            final_prompt = generate_strict_prompt(target_words)
            
            st.markdown("### 👇 点击代码块右上角复制")
            st.code(final_prompt, language="markdown")
            
            st.info("💡 复制后，发送给 ChatGPT/Claude。建议要求它生成可下载的 .csv 文件以便直接导入 Anki。")