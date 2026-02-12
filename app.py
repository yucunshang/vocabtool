import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import time

# ==========================================
# 0. 基础配置
# ==========================================
st.set_page_config(
    page_title="Vocab Master", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    
    /* 大按钮 */
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3.2em; font-weight: bold; font-size: 16px !important;
        margin-top: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 文本框优化 */
    .stTextArea textarea { font-size: 15px !important; border-radius: 10px; font-family: monospace; }
    
    /* 折叠栏样式 */
    [data-testid="stExpander"] { border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px; }
    
    /* 复制提示 */
    .copy-tip { font-size: 12px; color: #888; margin-bottom: 5px; }
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
            
            # 【关键修复】先按排名排序(升序)，然后去重保留第一个
            # 这样保证 say (rank 19) 优先于 say (rank 11771)
            df = df.sort_values(r_col, ascending=True)
            df_unique = df.drop_duplicates(subset=[w_col], keep='first')
            
            vocab_dict = pd.Series(df_unique[r_col].values, index=df_unique[w_col]).to_dict()
            return vocab_dict, df_unique, r_col, w_col
        except Exception as e:
            st.error(f"数据加载出错: {e}")
            return {}, None, None, None
    return {}, None, None, None

VOCAB_DICT, FULL_DF, RANK_COL, WORD_COL = load_data()
def get_lemma(word): return lemminflect.getLemma(word, upos='VERB')[0] 

# ==========================================
# 2. 核心逻辑
# ==========================================
def analyze_text(text, current_lvl, target_lvl):
    raw_words = re.findall(r"[a-z]+", text.lower())
    unique_words = set(raw_words)
    
    data_list = []
    for w in unique_words:
        if len(w) < 2: continue
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999)
        
        category = "Beyond"
        if rank <= current_lvl: category = "Mastered"
        elif rank <= target_lvl: category = "Target"
        
        data_list.append({"Word": lemma, "Rank": int(rank), "Category": category})
        
    df = pd.DataFrame(data_list)
    return df

def generate_prompt(word_list):
    """
    使用用户指定的特定格式生成 Prompt
    """
    word_str = ", ".join(word_list)
    
    prompt = f"""Role: High-Efficiency Anki Card Creator
Task: Convert the provided word list into a strict CSV/TXT code block.

--- OUTPUT FORMAT RULES ---

1. Structure: 2 Columns only. Comma-separated. All fields double-quoted.
   Format: "Front","Back"

2. Column 1 (Front):
   - Content: A natural, short English phrase or collocation containing the target word.
   - Style: Plain text (NO bolding, NO extra symbols).

3. Column 2 (Back):
   - Content: Definition + Example + Etymology.
   - HTML Layout: Definition<br><em>Example Sentence</em><br>【源】Etymology
   - Constraints:
     - Use single <br> tag for line breaks (to save tokens).
     - Example sentence must be wrapped in <em> tags (Italics).
     - Definition: Concise English.

4. Etymology Style (Crucial):
   - Language: CHINESE (中文).
   - Style: Logic-based, extremely concise. Use arrows (→). No narrative filler.
   - Format: 【源】Root/Origin (Meaning) → Result/Logic.
   - Example: 【源】Lat. 'vigere' (活跃) → 精力/活力

5. Atomicity Principle (Strict):
   - If a word has distinct meanings (e.g., Noun vs. Verb, Literal vs. Metaphorical), generate SEPARATE rows for each. Do not combine them.

6. Output Requirement:
   - Output the Code Block ONLY. No conversational text before or after.

--- EXAMPLE OUTPUT ---
"limestone quarry","deep pit for extracting stone<br><em>The company owns a granite quarry.</em><br>【源】古法语 quarriere (方石) → 切石场"
"hunter's quarry","animal pursued by a hunter<br><em>The eagle spotted its quarry.</em><br>【源】古法语 cuir (皮革) → 放皮上的内脏赏赐 → 猎物"
"stiffen with cold","make or become rigid<br><em>His muscles began to stiffen.</em><br>【源】stiff (僵硬) + -en (使动)"

--- MY WORD LIST ---
{word_str}
"""
    return prompt

# ==========================================
# 3. 主界面
# ==========================================
st.title("⚡️ Vocab Master")

if FULL_DF is None:
    st.error("⚠️ 缺少词频文件")
else:
    # --- 模式选择 ---
    mode = st.radio("模式", ["📖 文本提取", "🔢 词频刷词", "🛠️ 格式转换"], horizontal=True, label_visibility="collapsed")
    st.divider()

    # ------------------------------------------------
    # 模式 A: 文本提取
    # ------------------------------------------------
    if mode == "📖 文本提取":
        st.caption("分析文章，筛选重点词")
        
        # 1. 输入
        c_a, c_b = st.columns(2)
        with c_a: curr_lvl = st.number_input("当前水平", 4000, step=500)
        with c_b: targ_lvl = st.number_input("目标水平", 8000, step=500)
        
        inp_type = st.radio("Input", ["粘贴", "上传"], horizontal=True, label_visibility="collapsed")
        
        user_text = ""
        if inp_type == "粘贴":
            user_text = st.text_area("在此粘贴文本", height=100)
        else:
            up = st.file_uploader("上传 (TXT/PDF)", type=["txt","pdf"])
            if up:
                try:
                    if up.name.endswith('.txt'): user_text = up.getvalue().decode("utf-8")
                    else: 
                        import PyPDF2
                        r = PyPDF2.PdfReader(up)
                        user_text = " ".join([p.extract_text() for p in r.pages])
                except: st.error("读取失败")

        # 2. 分析按钮
        if user_text and st.button("🔍 开始分析", type="primary"):
            with st.spinner("分析中..."):
                t0 = time.time()
                df_res = analyze_text(user_text, curr_lvl, targ_lvl)
                st.session_state['analysis_df'] = df_res
                st.session_state['analysis_time'] = time.time() - t0
        
        # 3. 结果展示
        if 'analysis_df' in st.session_state:
            df = st.session_state['analysis_df']
            
            # 排序逻辑：重点词按 Rank 降序 (难->易)
            df_target = df[df['Category'] == 'Target'].sort_values(by="Rank", ascending=False)
            df_mastered = df[df['Category'] == 'Mastered'].sort_values(by="Rank")
            df_beyond = df[df['Category'] == 'Beyond'].sort_values(by="Rank")
            
            st.success(f"共 {len(df)} 词 (耗时 {st.session_state['analysis_time']:.2f}s)")
            
            t1, t2, t3 = st.tabs([
                f"🎯 重点 ({len(df_target)})", 
                f"✅ 已掌握 ({len(df_mastered)})", 
                f"🚀 超纲 ({len(df_beyond)})"
            ])
            
            # --- 重点词 Tab ---
            with t1:
                default_target_str = ", ".join(df_target["Word"].tolist())
                
                # 编辑区
                with st.expander("📝 编辑重点词 (可折叠)", expanded=True):
                    st.caption("👇 在此修改列表：")
                    edited_target_str = st.text_area("Target List", value=default_target_str, height=150, key="ta_target")
                
                # 纯单词列表一键复制
                st.markdown("<p class='copy-tip'>👇 纯单词列表 (点击右上角复制)</p>", unsafe_allow_html=True)
                st.code(edited_target_str, language="text")

                final_words = [w.strip() for w in edited_target_str.split(',') if w.strip()]
                
                if final_words:
                    # 🟢 分批逻辑 (100个一组)
                    BATCH_SIZE = 100
                    total = len(final_words)
                    
                    if total > BATCH_SIZE:
                        st.warning(f"单词较多 ({total})，自动分批 (每批 {BATCH_SIZE})")
                        num_batches = (total // BATCH_SIZE) + (1 if total % BATCH_SIZE != 0 else 0)
                        
                        sel_batch = st.radio(
                            "选择批次:", 
                            range(1, num_batches + 1), 
                            format_func=lambda x: f"第 {x} 批 ({min(x*BATCH_SIZE, total)}词)",
                            horizontal=True
                        )
                        
                        start = (sel_batch - 1) * BATCH_SIZE
                        batch_words = final_words[start : start + BATCH_SIZE]
                        
                        if st.button(f"🚀 生成 Prompt (第 {sel_batch} 批)", type="primary"):
                            prompt = generate_prompt(batch_words)
                            st.code(prompt, language="markdown")
                            st.success("👆 点击代码块右上角复制，发送给 AI")
                    else:
                        if st.button("🚀 生成 Prompt (全部)", type="primary"):
                            prompt = generate_prompt(final_words)
                            st.code(prompt, language="markdown")
                            st.success("👆 点击代码块右上角复制，发送给 AI")

            # --- 已掌握 Tab ---
            with t2:
                words_m = ", ".join(df_mastered["Word"].tolist())
                st.caption("👇 点击右上角复制")
                st.code(words_m, language="text")
            
            # --- 超纲 Tab ---
            with t3:
                words_b = ", ".join(df_beyond["Word"].tolist())
                st.caption("👇 点击右上角复制")
                st.code(words_b, language="text")

    # ------------------------------------------------
    # 模式 B: 刷词
    # ------------------------------------------------
    elif mode == "🔢 词频刷词":
        c1, c2 = st.columns(2)
        with c1: s_r = st.number_input("Start", 8000, step=50)
        with c2: cnt = st.number_input("Count", 50, step=10)
        
        if st.button("提取"):
            res = FULL_DF[FULL_DF[RANK_COL] >= s_r].sort_values(RANK_COL).head(cnt)
            w_str = ", ".join(res[WORD_COL].tolist())
            st.session_state['range_str'] = w_str
            
        if 'range_str' in st.session_state:
            with st.expander("📝 编辑列表", expanded=True):
                edited_range_str = st.text_area("List", value=st.session_state['range_str'], height=150)
            
            st.code(edited_range_str, language="text")
            
            words = [w.strip() for w in edited_range_str.split(',') if w.strip()]
            
            if st.button("🚀 生成 Prompt", type="primary"):
                prompt = generate_prompt(words)
                st.code(prompt, language="markdown")

    # ------------------------------------------------
    # 模式 C: 转换
    # ------------------------------------------------
    elif mode == "🛠️ 格式转换":
        st.markdown("### 📥 转 Anki CSV")
        st.caption("粘贴 AI 回复 (无表头)")
        txt = st.text_area("粘贴内容", height=200)
        
        if txt:
            # 简单的清洗，确保 AI 没带 Markdown 代码块标记
            clean_txt = txt.replace("```csv", "").replace("```", "").strip()
            st.download_button("📥 下载 .csv", clean_txt.encode("utf-8"), "anki.csv", "text/csv", type="primary")