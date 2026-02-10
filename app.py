import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import json

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode {
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 16px !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    div[role="radiogroup"] > label {
        font-weight: bold;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color-light);
        padding: 5px 15px;
        border-radius: 8px;
        margin-right: 10px;
    }
    div[role="radiogroup"] > label:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据加载 (Data Loading) - 核心优化
# ==========================================
@st.cache_data
def load_knowledge_base():
    """从 JSON 文件加载静态知识库，极大提升性能"""
    try:
        # 1. 术语库
        with open('data/terms.json', 'r', encoding='utf-8') as f:
            terms = json.load(f)
        # 2. 专有名词库
        with open('data/proper.json', 'r', encoding='utf-8') as f:
            proper = json.load(f)
        # 3. 补丁词库
        with open('data/patch.json', 'r', encoding='utf-8') as f:
            patch = json.load(f)
        # 4. 歧义词 (列表转集合)
        with open('data/ambiguous.json', 'r', encoding='utf-8') as f:
            ambiguous = set(json.load(f))
            
        # 确保术语 key 全小写，防止匹配失败
        terms = {k.lower(): v for k, v in terms.items()}
        proper = {k.lower(): v for k, v in proper.items()}
        
        return terms, proper, patch, ambiguous
    except FileNotFoundError:
        st.error("⚠️ 缺少数据文件！请确保 `data/` 文件夹下包含 terms.json, proper.json, patch.json, ambiguous.json")
        return {}, {}, {}, set()

# 全局变量加载
BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

# ==========================================
# 3. 初始化 NLP
# ==========================================
@st.cache_resource
def setup_nltk():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    for pkg in ['averaged_perceptron_tagger', 'punkt']:
        try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except: pass

setup_nltk()

def smart_lemmatize(text):
    words = re.findall(r"[a-zA-Z']+", text)
    results = []
    for w in words:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict:
            results.append(w.lower())
            continue
        if 'ADJ' in lemmas_dict: lemma = lemmas_dict['ADJ'][0]
        elif 'ADV' in lemmas_dict: lemma = lemmas_dict['ADV'][0]
        elif 'VERB' in lemmas_dict: lemma = lemmas_dict['VERB'][0]
        elif 'NOUN' in lemmas_dict: lemma = lemmas_dict['NOUN'][0]
        else: lemma = list(lemmas_dict.values())[0][0]
        results.append(lemma)
    return " ".join(results)

# ==========================================
# 4. 词库加载 (CSV)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    vocab = {}
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
            df = df.sort_values(r_col, ascending=True)
            df = df.drop_duplicates(subset=[w_col], keep='first')
            vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except: pass
    
    # 注入 JSON 加载的补丁
    for word, rank in BUILTIN_PATCH_VOCAB.items():
        if word not in vocab: vocab[word] = rank
        else:
            if vocab[word] > 20000: vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 5. AI 指令生成器
# ==========================================
def generate_ai_prompt(word_list, output_format, def_mode="single", is_term_list=False):
    words_str = ", ".join(word_list)
    
    definition_instruction = ""
    if is_term_list or def_mode == "term":
        definition_instruction = "- **领域锁定**：单词带有 (Domain) 标签，**必须**仅提供符合该领域背景的专业释义。"
    elif def_mode == "split":
        definition_instruction = """- **熟词深挖 (Polymsey Splitting)**：这些是高频常用词，为了掌握其不同用法，**请将不同的含义拆分为多条独立的数据（多张卡片）**。
    - 例如 'fair' 应拆分为：
      1. fair (adj) - reasonable/impartial (公平的)
      2. fair (n) - gathering/market (集市)
    - 不要把所有意思挤在一张卡片里。"""
    else: # single
        definition_instruction = "- **极简速记 (Minimalist)**：这些是生词，请**仅提供 1 个最核心、最常用的释义**。严禁罗列多个义项，减轻记忆负担。"

    if output_format == 'csv':
        format_req = "CSV Code Block (后缀名 .csv)"
        format_desc = "请直接输出标准 CSV 代码块。"
    else:
        format_req = "TXT Code Block (后缀名 .txt)"
        format_desc = "请输出纯文本 TXT 代码块。"

    prompt = f"""
请扮演一位专业的 Anki 制卡专家。这是我整理的单词列表，请严格按照以下【释义策略】为我生成导入文件。

1. 核心原则：释义策略
{definition_instruction}

2. 卡片正面 (Column 1: Front)
- 内容：提供自然的短语或搭配 (Phrase/Collocation)。
- 样式：纯文本。

3. 卡片背面 (Column 2: Back)
- 格式：HTML 排版，包含三部分，必须使用 <br><br> 分隔。
- 结构：英文释义<br><br><em>斜体例句</em><br><br>【词源/词根词缀】中文助记 (词源优先)

4. 输出格式标准 ({format_req})
- {format_desc}
- 关键格式：使用英文逗号 (,) 分隔，且每个字段内容必须用英文双引号 ("...") 包裹。

待处理单词：
{words_str}
"""
    return prompt

# ==========================================
# 6. 通用分析函数
# ==========================================
def analyze_text(raw_text, mode="auto"):
    raw_items = []
    if "按行" in mode:
        lines = raw_text.split('\n')
        for line in lines:
            if line.strip(): raw_items.append(line.strip())
    else:
        clean_text = re.sub(r'[,.\n\t]', ' ', raw_text)
        raw_items = clean_text.split()
    
    seen = set()
    unique_items = [] 
    JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're'}
    
    for item in raw_items:
        item_cleaned = item.strip()
        item_lower = item_cleaned.lower()
        
        if item_lower in seen: continue
        if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
        if item_lower in JUNK_WORDS: continue
        
        # 1. 术语身份
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            unique_items.append({
                "word": f"{item_cleaned} ({domain})", 
                "rank": 0,
                "cat": "term",
                "raw": item_lower
            })
        
        # 2. 专名身份 (Rank 1, 方便过滤)
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_cleaned.title())
            unique_items.append({
                "word": display,
                "rank": 1, 
                "cat": "proper",
                "raw": item_lower
            })
            
        # 3. 普通身份
        rank = vocab_dict.get(item_lower, 99999)
        if rank != 99999:
            unique_items.append({
                "word": item_cleaned,
                "rank": rank,
                "cat": "general",
                "raw": item_lower
            })
        
        seen.add(item_lower)
        
    return pd.DataFrame(unique_items)

# ==========================================
# 7. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro")

app_mode = st.radio("选择功能模式:", 
    ["🛠️ 智能还原", "📊 单词分级 (全量)", "🎯 智能精选 (Top N)"], 
    horizontal=True
)
st.divider()

# ---------------------------------------------------------
# 模式 A: 智能还原
# ---------------------------------------------------------
if "智能还原" in app_mode:
    c1, c2 = st.columns(2)
    with c1:
        raw_text = st.text_area("输入原始文章", height=400, placeholder="He was excited.")
        if st.button("开始还原", type="primary"):
            res = smart_lemmatize(raw_text)
            st.code(res, language='text')
            st.caption("👆 一键复制")

# ---------------------------------------------------------
# 模式 B: 单词分级 (全量)
# ---------------------------------------------------------
elif "单词分级" in app_mode:
    col_level1, col_level2, _ = st.columns([1, 1, 2])
    with col_level1: current_level = st.number_input("当前水平", 0, 30000, 9000, 500)
    with col_level2: target_level = st.number_input("目标水平", 0, 30000, 15000, 500)
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词", "按行处理"), horizontal=True)
        grade_input = st.text_area("input_box", height=400, placeholder="motion\nenergy\nrun\nset", label_visibility="collapsed")
        btn_grade = st.button("开始分级", type="primary", use_container_width=True)

    with g_col2:
        if btn_grade and grade_input and vocab_dict:
            df = analyze_text(grade_input, input_mode)
            if not df.empty:
                def categorize(row):
                    if row['cat'] == 'term': return 'term'
                    if row['cat'] == 'proper': return 'proper'
                    r = row['rank']
                    if r <= current_level: return "known"
                    elif r <= target_level: return "target"
                    else: return "beyond"
                
                df['final_cat'] = df.apply(categorize, axis=1)
                df = df.sort_values(by='rank')

                t1, t2, t3, t4, t5 = st.tabs(["🟣 专业术语", "🟡 重点", "🔵 专有名词", "🔴 超纲", "🟢 已掌握"])
                
                def render_tab(tab_obj, cat_key, label, def_mode):
                    with tab_obj:
                        sub = df[df['final_cat'] == cat_key]
                        st.caption(f"共 {len(sub)} 个")
                        if not sub.empty:
                            words = sub['word'].tolist()
                            with st.expander("👁️ 查看列表", expanded=False): st.code("\n".join(words))
                            
                            st.markdown(f"**🤖 AI 指令 ({label})**")
                            has_term = (cat_key == 'term')
                            
                            p_csv = generate_ai_prompt(words, 'csv', def_mode, is_term_list=has_term)
                            p_txt = generate_ai_prompt(words, 'txt', def_mode, is_term_list=has_term)
                            
                            t_csv, t_txt = st.tabs(["📋 CSV 指令", "📝 TXT 指令"])
                            with t_csv: st.code(p_csv, language='markdown')
                            with t_txt: st.code(p_txt, language='markdown')
                        else: st.info("无")

                render_tab(t1, "term", "术语", def_mode="term")   
                render_tab(t2, "target", "重点", def_mode="single") 
                render_tab(t3, "proper", "专名", def_mode="single")
                render_tab(t4, "beyond", "超纲", def_mode="single") 
                render_tab(t5, "known", "熟词", def_mode="split")  

# ---------------------------------------------------------
# 模式 C: 智能精选 (Top N)
# ---------------------------------------------------------
elif "Top N" in app_mode:
    st.info("💡 此模式自动过滤简单词，按 **由易到难** 挑选。")
    
    c_set1, c_set2, c_set3 = st.columns([1, 1, 1])
    with c_set1: top_n = st.number_input("🎯 筛选数量", 10, 500, 50, 10)
    with c_set2: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 3000, 500)
    with c_set3: st.write("") 
        
    c_input, c_btn = st.columns([3, 1])
    with c_input:
        topn_input = st.text_area("输入", height=150, placeholder="motion\nenergy\nrun", label_visibility="collapsed")
    with c_btn:
        btn_topn = st.button("🎲 生成精选", type="primary", use_container_width=True)

    if btn_topn and topn_input and vocab_dict:
        df = analyze_text(topn_input, "自动分词") 
        
        if not df.empty:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce').fillna(99999)
            
            term_mask = (df['cat'] == 'term')
            general_mask = (df['cat'].isin(['general', 'proper'])) & (df['rank'] >= min_rank_threshold)
            
            valid_candidates = df[term_mask | general_mask].copy()
            sorted_df = valid_candidates.sort_values(by='rank', ascending=True)
            top_df = sorted_df.head(top_n)
            
            all_ids = set(df.index)
            top_ids = set(top_df.index)
            rest_ids = all_ids - top_ids
            rest_df = df.loc[list(rest_ids)].sort_values(by='rank')
            
            st.divider()
            col_win, col_rest = st.columns(2)
            
            # === 左栏 ===
            with col_win:
                st.success(f"🔥 精选 Top {len(top_df)}")
                if not top_df.empty:
                    words = top_df['word'].tolist()
                    with st.expander("列表", expanded=True): st.code("\n".join(words))
                    
                    st.markdown("**🤖 AI 指令 (核心单义)**")
                    has_term = any('(' in w for w in words)
                    mode = "single" if not has_term else "term"
                    
                    p_csv = generate_ai_prompt(words, 'csv', mode, is_term_list=has_term)
                    p_txt = generate_ai_prompt(words, 'txt', mode, is_term_list=has_term)
                    
                    t1, t2 = st.tabs(["CSV", "TXT"])
                    with t1: st.code(p_csv, language='markdown')
                    with t2: st.code(p_txt, language='markdown')
                else: st.warning("无")

            # === 右栏 ===
            with col_rest:
                st.subheader(f"💤 剩余 {len(rest_df)} 个")
                if not rest_df.empty:
                    words_rest = rest_df['word'].tolist()
                    with st.expander("列表", expanded=False): st.code("\n".join(words_rest))
                    
                    st.markdown("**🤖 AI 指令 (备用)**")
                    p_csv_r = generate_ai_prompt(words_rest, 'csv', "single")
                    p_txt_r = generate_ai_prompt(words_rest, 'txt', "single")
                    
                    rt1, rt2 = st.tabs(["CSV", "TXT"])
                    with rt1: st.code(p_csv_r, language='markdown')
                    with rt2: st.code(p_txt_r, language='markdown')