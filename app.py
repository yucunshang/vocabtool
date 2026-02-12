import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import time

# ==========================================
# 0. 基础配置 (回归 Centered 布局)
# ==========================================
st.set_page_config(
    page_title="Vocab Master", 
    page_icon="⚡️", 
    layout="centered", # 手机端最佳布局
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* 界面紧凑优化 */
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    
    /* 按钮大尺寸，适合手指 */
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3.2em; font-weight: bold; font-size: 16px !important;
        margin-top: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 表格样式 */
    [data-testid="stDataFrameResizable"] { border: 1px solid #ddd; border-radius: 8px; }
    
    /* 设置栏样式 */
    [data-testid="stExpander"] { border-radius: 10px; border: 1px solid #ddd; margin-bottom: 15px; }
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
            # 默认按 Rank 升序
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

def generate_prompt(word_list, settings):
    word_str = ", ".join(word_list)
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

--- WORD LIST ({len(word_list)} words) ---
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
    # --- 顶栏设置 (折叠) ---
    with st.expander("⚙️ 全局设置 (Prompt Settings)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            set_format = st.selectbox("格式", ["CSV", "TXT"], index=0)
            set_lang = st.selectbox("语言", ["Chinese", "English"], index=0)
        with c2:
            set_ex_count = st.number_input("例句数", 1, 3, 1)
            # 无需 case 选择，已在 Prompt 写死为 Phrase
            
    settings = {"format": set_format, "lang": set_lang, "example_count": set_ex_count}

    # --- 模式选择 ---
    mode = st.radio("模式", ["📖 文本提取", "🔢 词频刷词", "🛠️ 格式转换"], horizontal=True, label_visibility="collapsed")
    st.divider()

    # ------------------------------------------------
    # 模式 A: 文本提取
    # ------------------------------------------------
    if mode == "📖 文本提取":
        st.caption("分析文章，筛选重点词")
        
        # 1. 输入区
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

        # 2. 分析按钮 (回归主界面)
        if user_text and st.button("🔍 开始分析", type="primary"):
            with st.spinner("分析中..."):
                t0 = time.time()
                df_res = analyze_text(user_text, curr_lvl, targ_lvl)
                st.session_state['analysis_df'] = df_res
                st.session_state['analysis_time'] = time.time() - t0
        
        # 3. 结果展示区
        if 'analysis_df' in st.session_state:
            df = st.session_state['analysis_df']
            
            # 筛选与排序
            df_target = df[df['Category'] == 'Target'].sort_values(by="Rank", ascending=False) # 重点词：难 -> 易
            df_mastered = df[df['Category'] == 'Mastered'].sort_values(by="Rank")
            df_beyond = df[df['Category'] == 'Beyond'].sort_values(by="Rank")
            
            st.success(f"共 {len(df)} 词 (耗时 {st.session_state['analysis_time']:.2f}s)")
            
            t1, t2, t3 = st.tabs([
                f"🎯 重点 ({len(df_target)})", 
                f"✅ 已掌握 ({len(df_mastered)})", 
                f"🚀 超纲 ({len(df_beyond)})"
            ])
            
            # --- 重点词 (可编辑) ---
            with t1:
                st.caption("👇 可直接修改单词，或在末尾添加。勾选并按 Del 删除。")
                edited_df = st.data_editor(
                    df_target[["Word", "Rank"]],
                    num_rows="dynamic",
                    use_container_width=True,
                    key="editor_target",
                    column_config={"Rank": st.column_config.NumberColumn("Rank")}
                )
                
                final_words = [str(w).strip() for w in edited_df["Word"].tolist() if str(w).strip()]
                
                if final_words:
                    st.divider()
                    
                    # 🟢 分批逻辑：200个一组
                    BATCH_SIZE = 200
                    total = len(final_words)
                    
                    if total > BATCH_SIZE:
                        st.warning(f"单词较多 ({total})，已自动分批 (每批 {BATCH_SIZE})")
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
                            prompt = generate_prompt(batch_words, settings)
                            st.code(prompt, language="markdown")
                    else:
                        if st.button("🚀 生成 Prompt (全部)", type="primary"):
                            prompt = generate_prompt(final_words, settings)
                            st.code(prompt, language="markdown")

            # --- 其他 (只读) ---
            with t2: st.code(", ".join(df_mastered["Word"]), language="text")
            with t3: st.code(", ".join(df_beyond["Word"]), language="text")

    # ------------------------------------------------
    # 模式 B: 刷词
    # ------------------------------------------------
    elif mode == "🔢 词频刷词":
        c1, c2 = st.columns(2)
        with c1: s_r = st.number_input("Start", 8000, step=50)
        with c2: cnt = st.number_input("Count", 50, step=10)
        
        if st.button("提取"):
            res = FULL_DF[FULL_DF[RANK_COL] >= s_r].sort_values(RANK_COL).head(cnt)
            st.session_state['range_df'] = res[[WORD_COL, RANK_COL]]
            
        if 'range_df' in st.session_state:
            st.caption("👇 可编辑列表")
            ed_df = st.data_editor(st.session_state['range_df'], num_rows="dynamic", use_container_width=True)
            words = [str(w).strip() for w in ed_df[WORD_COL] if str(w).strip()]
            
            if st.button("🚀 生成 Prompt", type="primary"):
                prompt = generate_prompt(words, settings)
                st.code(prompt, language="markdown")

    # ------------------------------------------------
    # 模式 C: 转换
    # ------------------------------------------------
    elif mode == "🛠️ 格式转换":
        st.markdown("### 📥 转 Anki CSV")
        txt = st.text_area("粘贴 AI 回复", height=200)
        if txt:
            st.download_button("📥 下载 .csv", txt.encode("utf-8"), "anki.csv", "text/csv", type="primary")