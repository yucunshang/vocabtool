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
    page_title="Vocab Master Pro", 
    page_icon="⚡️", 
    layout="wide", #以此容纳表格
    initial_sidebar_state="expanded" # 默认展开侧边栏以便看到按钮
)

st.markdown("""
<style>
    /* 界面优化 */
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* 侧边栏优化 */
    [data-testid="stSidebar"] { background-color: #f9f9f9; }
    
    /* 按钮大尺寸 */
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em; font-weight: bold; font-size: 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 表格编辑器优化 */
    [data-testid="stDataFrameResizable"] { border: 1px solid #ddd; border-radius: 8px; }
    
    /* 提示框 */
    .stAlert { border-radius: 8px; }
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
            # 默认按 Rank 升序 (1, 2, 3...)
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
    """分析文本，返回 DataFrame 以便编辑器使用"""
    raw_words = re.findall(r"[a-z]+", text.lower())
    unique_words = set(raw_words)
    
    data_list = [] # 存放字典 {'Word': w, 'Rank': r, 'Category': c}
    
    for w in unique_words:
        if len(w) < 2: continue
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999) # 99999 代表未收录
        
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
# 3. 侧边栏：全局控制 (需求1：按钮一直在)
# ==========================================
with st.sidebar:
    st.header("🎛️ 控制台")
    
    # 模式选择
    mode = st.radio("模式", ["📖 文本提取", "🔢 词频刷词", "🛠️ 格式转换"])
    st.divider()
    
    # 全局 Prompt 设置
    with st.expander("⚙️ 生成设置", expanded=False):
        set_format = st.selectbox("格式", ["CSV", "TXT"], index=0)
        set_lang = st.selectbox("语言", ["Chinese", "English"], index=0)
        set_ex_count = st.number_input("例句数", 1, 3, 1)
    
    settings = {"format": set_format, "lang": set_lang, "example_count": set_ex_count}
    
    # --- 提取模式的输入 ---
    if mode == "📖 文本提取":
        st.subheader("1. 输入文本")
        curr_lvl = st.number_input("当前水平", 4000, step=500)
        targ_lvl = st.number_input("目标水平", 8000, step=500)
        
        inp_type = st.radio("来源", ["粘贴", "文件"], horizontal=True)
        
        user_text = ""
        if inp_type == "粘贴":
            user_text = st.text_area("在此粘贴", height=150)
        else:
            up = st.file_uploader("上传 (TXT/PDF)", type=["txt","pdf"])
            if up:
                try:
                    if up.name.endswith('.txt'): user_text = up.getvalue().decode("utf-8")
                    else: 
                        import PyPDF2
                        r = PyPDF2.PdfReader(up)
                        user_text = " ".join([p.extract_text() for p in r.pages])
                except: st.error("文件读取失败")
        
        # 🟢 需求1：分析按钮放在 Sidebar，永远可见
        analyze_clicked = st.button("🔍 开始分析", type="primary")

# ==========================================
# 4. 主界面：结果展示与编辑
# ==========================================
st.title("⚡️ Vocab Master Pro")

if FULL_DF is None:
    st.error("⚠️ 缺少词频文件")
else:
    # ------------------------------------------------
    # 模式 A: 文本提取 (核心升级)
    # ------------------------------------------------
    if mode == "📖 文本提取":
        # 使用 Session State 保存分析结果，防止刷新丢失
        if analyze_clicked and user_text:
            with st.spinner("正在极速分析..."):
                t0 = time.time()
                df_result = analyze_text(user_text, curr_lvl, targ_lvl)
                st.session_state['analysis_df'] = df_result
                st.session_state['analysis_time'] = time.time() - t0
        
        if 'analysis_df' in st.session_state:
            df = st.session_state['analysis_df']
            t_taken = st.session_state.get('analysis_time', 0)
            
            st.success(f"✅ 分析完成！共 {len(df)} 词 (耗时 {t_taken:.2f}s)")
            
            # 分类筛选
            df_target = df[df['Category'] == 'Target'].copy()
            df_mastered = df[df['Category'] == 'Mastered'].copy()
            df_beyond = df[df['Category'] == 'Beyond'].copy()
            
            # 🟢 需求3：重点词按 Rank 从高到低 (难 -> 易)
            df_target = df_target.sort_values(by="Rank", ascending=False)
            
            tab1, tab2, tab3 = st.tabs([
                f"🎯 重点 ({len(df_target)})", 
                f"✅ 已掌握 ({len(df_mastered)})", 
                f"🚀 超纲 ({len(df_beyond)})"
            ])
            
            # --- 重点词 Tab (可编辑 + 分批) ---
            with tab1:
                st.markdown("### 📝 编辑重点词列表")
                st.caption("提示：你可以直接修改单词，或在最后一行添加新词。勾选左侧复选框并按 Delete 可删除行。")
                
                # 🟢 需求2：可编辑表格 (Data Editor)
                # num_rows="dynamic" 允许添加/删除行
                edited_df = st.data_editor(
                    df_target[["Word", "Rank"]],
                    num_rows="dynamic",
                    use_container_width=True,
                    key="editor_target",
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank (越大越生僻)")
                    }
                )
                
                # 获取编辑后的最终列表
                final_words = edited_df["Word"].tolist()
                final_words = [str(w).strip() for w in final_words if str(w).strip()] # 清洗空行
                
                if final_words:
                    st.divider()
                    st.markdown("### 🚀 生成 AI 指令")
                    
                    # 🟢 需求4：分批处理逻辑
                    BATCH_SIZE = 30  # 建议批次大小
                    total_words = len(final_words)
                    
                    if total_words > BATCH_SIZE:
                        st.warning(f"⚠️ 单词总数 ({total_words}) 较多，建议分批生成以保证 AI 输出质量。")
                        
                        # 计算批次数
                        num_batches = (total_words // BATCH_SIZE) + (1 if total_words % BATCH_SIZE != 0 else 0)
                        
                        # 批次选择器
                        selected_batch = st.radio(
                            "选择批次:",
                            options=range(1, num_batches + 1),
                            format_func=lambda x: f"第 {x} 批 (单词 {(x-1)*BATCH_SIZE + 1} - {min(x*BATCH_SIZE, total_words)})",
                            horizontal=True
                        )
                        
                        # 切片
                        start_idx = (selected_batch - 1) * BATCH_SIZE
                        end_idx = start_idx + BATCH_SIZE
                        batch_words = final_words[start_idx : end_idx]
                        
                        st.info(f"当前选中: **{len(batch_words)}** 个单词")
                        
                        if st.button(f"生成 Prompt (第 {selected_batch} 批)", type="primary"):
                            prompt = generate_prompt(batch_words, settings)
                            st.code(prompt, language="markdown")
                            
                    else:
                        # 数量少，直接生成
                        if st.button("生成 Prompt (全部)", type="primary"):
                            prompt = generate_prompt(final_words, settings)
                            st.code(prompt, language="markdown")

            # --- 其他 Tab (仅展示可复制) ---
            with tab2:
                st.code(", ".join(df_mastered["Word"].tolist()), language="text")
            with tab3:
                st.code(", ".join(df_beyond["Word"].tolist()), language="text")

    # ------------------------------------------------
    # 模式 B: 刷词
    # ------------------------------------------------
    elif mode == "🔢 词频刷词":
        c1, c2 = st.columns(2)
        with c1: s_r = st.number_input("Start", 8000, step=50)
        with c2: cnt = st.number_input("Count", 50, step=10)
        
        if st.button("提取单词"):
            res = FULL_DF[FULL_DF[RANK_COL] >= s_r].sort_values(RANK_COL).head(cnt)
            # 存入 session 用于编辑
            st.session_state['range_df'] = res[[WORD_COL, RANK_COL]]
            
        if 'range_df' in st.session_state:
            st.markdown("### 📝 单词列表 (可编辑)")
            
            # 同样使用 Editor
            range_edited = st.data_editor(
                st.session_state['range_df'],
                num_rows="dynamic",
                use_container_width=True,
                key="editor_range"
            )
            
            words_to_gen = range_edited[WORD_COL].tolist()
            
            if st.button("生成 Prompt"):
                prompt = generate_prompt(words_to_gen, settings)
                st.code(prompt, language="markdown")

    # ------------------------------------------------
    # 模式 C: 转换
    # ------------------------------------------------
    elif mode == "🛠️ 格式转换":
        st.markdown("### 📥 转 Anki CSV")
        txt = st.text_area("粘贴 AI 回复", height=200)
        if txt:
            st.download_button("📥 下载 .csv", txt.encode("utf-8"), "anki.csv", "text/csv", type="primary")