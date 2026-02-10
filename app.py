import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk

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
    div[role="radiogroup"] > label {
        font-weight: bold;
        background-color: #f0f2f6;
        padding: 0 15px;
        border-radius: 5px;
    }
    /* 隐藏侧边栏按钮 */
    [data-testid="stSidebarCollapsedControl"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 内置词库 (v47 双重身份版数据)
# ==========================================
BUILTIN_TECHNICAL_TERMS = {
    # 用户指定补充
    "metal": "Chem", "motion": "Law", "gravity": "Phys", "molecule": "Chem",
    "vacuum": "Phys", "electron": "Phys", "quantum": "Phys", "velocity": "Phys",
    "friction": "Phys", "catalyst": "Chem", "equilibrium": "Chem",
    
    # CS (示例)
    "algorithm": "CS", "recursion": "CS", "latency": "CS", "throughput": "CS", 
    "api": "CS", "json": "CS", "backend": "CS", "frontend": "CS", "fullstack": "CS",
    "neural": "AI", "transformer": "AI", "embedding": "AI", "inference": "AI",
    "python": "CS", "java": "CS", "docker": "CS", "kubernetes": "CS", "linux": "CS",
    
    # Math
    "derivative": "Math", "integral": "Math", "matrix": "Math", "vector": "Math",
    "theorem": "Math", "variance": "Math", "deviation": "Math", "correlation": "Math",
    
    # Phys
    "acceleration": "Phys", "momentum": "Phys", "inertia": "Phys", "thermodynamics": "Phys",
    "entropy": "Phys", "enthalpy": "Phys", "kinetic": "Phys", "photon": "Phys",
    
    # Chem
    "compound": "Chem", "solvent": "Chem", "solute": "Chem", "concentration": "Chem",
    "alkali": "Chem", "covalent": "Chem", "ionic": "Chem", "oxidation": "Chem",
    
    # Bio
    "mitochondria": "Bio", "ribosome": "Bio", "membrane": "Bio", "cytoplasm": "Bio",
    "dna": "Bio", "rna": "Bio", "chromosome": "Bio", "genome": "Bio",
    
    # Biz
    "revenue": "Biz", "margin": "Biz", "liability": "Biz", "equity": "Biz", "dividend": "Biz",
    "audit": "Biz", "fiscal": "Biz", "inflation": "Econ", "deflation": "Econ",
    
    # Law
    "plaintiff": "Law", "defendant": "Law", "verdict": "Law", "prosecutor": "Law",
    "tort": "Law", "felony": "Law", "affidavit": "Law", "subpoena": "Law"
}
BUILTIN_TECHNICAL_TERMS = {k.lower(): v for k, v in BUILTIN_TECHNICAL_TERMS.items()}

PROPER_NOUNS_DB = {
    "usa": "USA", "uk": "UK", "uae": "UAE", "prc": "PRC",
    "america": "America", "england": "England", "scotland": "Scotland", "wales": "Wales",
    "japan": "Japan", "korea": "Korea", "france": "France", "germany": "Germany", "italy": "Italy",
    "spain": "Spain", "russia": "Russia", "india": "India", "brazil": "Brazil", "canada": "Canada",
    "australia": "Australia", "mexico": "Mexico", "egypt": "Egypt", "china": "China",
    "switzerland": "Switzerland", "sweden": "Sweden", "norway": "Norway", "denmark": "Denmark",
    "finland": "Finland", "netherlands": "Netherlands", "belgium": "Belgium", "austria": "Austria",
    "greece": "Greece", "turkey": "Turkey", "israel": "Israel", "saudi arabia": "Saudi Arabia",
    "singapore": "Singapore", "malaysia": "Malaysia", "thailand": "Thailand", "vietnam": "Vietnam",
    "indonesia": "Indonesia", "philippines": "Philippines",
    "london": "London", "paris": "Paris", "tokyo": "Tokyo", "beijing": "Beijing",
    "shanghai": "Shanghai", "hong kong": "Hong Kong", "sydney": "Sydney", 
    "melbourne": "Melbourne", "berlin": "Berlin", "rome": "Rome", "madrid": "Madrid",
    "new york": "New York", "los angeles": "Los Angeles", "san francisco": "San Francisco",
    "chicago": "Chicago", "seattle": "Seattle", "boston": "Boston", "houston": "Houston",
    "moscow": "Moscow", "cairo": "Cairo", "dubai": "Dubai", "mumbai": "Mumbai",
    "africa": "Africa", "asia": "Asia", "europe": "Europe", "antarctica": "Antarctica",
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
    "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday",
    "january": "January", "february": "February", "march": "March", "april": "April", 
    "may": "May", "june": "June", "july": "July", "august": "August", 
    "september": "September", "october": "October", "november": "November", "december": "December",
    "christmas": "Christmas", "easter": "Easter", "thanksgiving": "Thanksgiving", "halloween": "Halloween",
    "google": "Google", "apple": "Apple", "microsoft": "Microsoft", "tesla": "Tesla",
    "amazon": "Amazon", "facebook": "Facebook", "twitter": "Twitter", "youtube": "YouTube", "instagram": "Instagram",
    "tiktok": "TikTok", "netflix": "Netflix", "spotify": "Spotify", "zoom": "Zoom",
    "nasa": "NASA", "fbi": "FBI", "cia": "CIA", "un": "UN", "eu": "EU", "nato": "NATO", "wto": "WTO", "who": "WHO",
    "iphone": "iPhone", "ipad": "iPad", "mac": "Mac", "windows": "Windows", "android": "Android",
    "wifi": "Wi-Fi", "internet": "Internet", "bluetooth": "Bluetooth",
    "mr": "Mr.", "mrs": "Mrs.", "ms": "Ms.", "dr": "Dr.", "prof": "Prof.",
    "phd": "PhD", "mba": "MBA", "ceo": "CEO", "cfo": "CFO", "cto": "CTO", "vip": "VIP"
}

BUILTIN_PATCH_VOCAB = {
    "online": 2000, "website": 2500, "app": 3000, "user": 1500, "data": 1000,
    "software": 3000, "hardware": 4000, "network": 2500, "server": 3500,
    "cloud": 3000, "algorithm": 6000, "database": 5000, "interface": 5000,
    "digital": 3000, "virtual": 4000, "smart": 2000, "mobile": 2500,
    "email": 2000, "text": 1000, "chat": 2000, "video": 1500, "audio": 3000,
    "link": 2000, "click": 2000, "search": 1500, "share": 1500, "post": 1500,
    "analysis": 2500, "strategy": 2500, "method": 2000, "theory": 2500,
    "research": 1500, "evidence": 2000, "significant": 2000, "factor": 1500,
    "process": 1000, "system": 1000, "available": 1500, "similar": 1500,
    "specific": 2000, "issue": 1000, "policy": 1500, "community": 1500,
    "development": 1500, "economic": 2000, "global": 2500, "environment": 2000,
    "challenge": 2500, "opportunity": 2000, "solution": 2500, "management": 2500,
    "okay": 500, "hey": 500, "yeah": 500, "wow": 1000, "cool": 1500,
    "super": 2000, "extra": 2500, "plus": 2000
}

AMBIGUOUS_WORDS = {
    "china", "turkey", "march", "may", "august", "polish"
}

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
# 4. 词库加载
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
    
    BUILTIN_PATCH_VOCAB = {
        "online": 2000, "website": 2500, "app": 3000, "user": 1500, "data": 1000,
        "software": 3000, "hardware": 4000, "network": 2500, "server": 3500,
        "cloud": 3000, "algorithm": 6000, "database": 5000, "interface": 5000,
        "analysis": 2500, "strategy": 2500, "method": 2000, "theory": 2500,
        "research": 1500, "evidence": 2000, "significant": 2000, "factor": 1500
    }
    for word, rank in BUILTIN_PATCH_VOCAB.items():
        if word not in vocab: vocab[word] = rank
        else:
            if vocab[word] > 20000: vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 5. AI 指令生成器
# ==========================================
def generate_ai_prompt(word_list, output_format, is_term_list=False):
    words_str = ", ".join(word_list)
    
    context_instruction = ""
    if is_term_list:
        context_instruction = "\n- 注意：这些单词是【带领域标签的专业术语 (e.g. word (Domain))】。**英文释义**请务必根据括号内的领域（如 Math, CS）提供该领域的精确释义。**中文解析**部分请优先拆解【词源、词根、词缀】以辅助记忆。"

    if output_format == 'csv':
        format_req = "CSV Code Block (后缀名 .csv)"
        format_desc = "请直接输出标准 CSV 代码块。"
    else:
        format_req = "TXT Code Block (后缀名 .txt)"
        format_desc = "请输出纯文本 TXT 代码块。"

    prompt = f"""
请扮演一位专业的 Anki 制卡专家。这是我整理的单词列表{context_instruction}，请严格按照以下【终极制卡标准】为我生成导入文件。

1. 核心原则：原子性 (Atomicity)
- 含义拆分：若单词有多个不同含义，拆分为多条数据。
- 严禁堆砌：每张卡片只承载一个特定语境下的含义。
- **领域匹配**：如果单词带有 (Domain) 标签，解释必须符合该领域背景。

2. 卡片正面 (Column 1: Front)
- 内容：提供自然的短语或搭配 (Phrase/Collocation)。
- 样式：纯文本。

3. 卡片背面 (Column 2: Back)
- 格式：HTML 排版，包含三部分，必须使用 <br><br> 分隔。
- 结构：英文释义<br><br><em>斜体例句</em><br><br>【词源/词根词缀】中文助记

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
        raw_items = raw_text.split()
    
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
        
        # 2. 专名身份
        if item_lower in PROPER_NOUNS_DB:
            unique_items.append({
                "word": PROPER_NOUNS_DB[item_lower],
                "rank": 0, 
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
        grade_input = st.text_area("input_box", height=400, placeholder="motion\nenergy", label_visibility="collapsed")
        btn_grade = st.button("开始分级", type="primary", use_container_width=True)

    with g_col2:
        if btn_grade and grade_input and vocab_dict:
            df = analyze_text(grade_input, input_mode)
            if not df.empty:
                def categorize(row):
                    if row['cat'] in ['term', 'proper']: return row['cat']
                    r = row['rank']
                    if r <= current_level: return "known"
                    elif r <= target_level: return "target"
                    else: return "beyond"
                
                df['final_cat'] = df.apply(categorize, axis=1)
                df = df.sort_values(by='rank')

                t1, t2, t3, t4, t5 = st.tabs(["🟣 专业术语", "🟡 重点", "🔵 专有名词", "🔴 超纲", "🟢 已掌握"])
                
                def render_tab(tab_obj, cat_key, label, is_term=False):
                    with tab_obj:
                        sub = df[df['final_cat'] == cat_key]
                        st.caption(f"共 {len(sub)} 个")
                        if not sub.empty:
                            words = sub['word'].tolist()
                            with st.expander("👁️ 查看列表", expanded=False): st.code("\n".join(words))
                            st.markdown(f"**🤖 AI 指令 ({label})**")
                            
                            p_csv = generate_ai_prompt(words, 'csv', is_term)
                            p_txt = generate_ai_prompt(words, 'txt', is_term)
                            
                            t_csv, t_txt = st.tabs(["📋 CSV 指令", "📝 TXT 指令"])
                            with t_csv: st.code(p_csv, language='markdown')
                            with t_txt: st.code(p_txt, language='markdown')
                        else: st.info("无")

                render_tab(t1, "term", "术语", True)
                render_tab(t2, "target", "重点", False)
                render_tab(t3, "proper", "专名", False)
                render_tab(t4, "beyond", "超纲", False)
                render_tab(t5, "known", "熟词", False)

# ---------------------------------------------------------
# 模式 C: 智能精选 (Top N)
# ---------------------------------------------------------
elif "Top N" in app_mode:
    st.info("💡 此模式自动过滤掉 **前 2000 个高频词** (太简单的)，然后从剩下的词中，按 **由易到难** 挑选出前 N 个。")
    
    c_set, c_input = st.columns([1, 2])
    with c_set:
        top_n = st.number_input("🎯 筛选数量 (N)", 10, 500, 50, 10)
        btn_topn = st.button("🎲 生成精选词单", type="primary", use_container_width=True)
    with c_input:
        topn_input = st.text_area("输入单词/文章", height=150, placeholder="motion\nenergy\nrevenue\n...", label_visibility="collapsed")

    if btn_topn and topn_input and vocab_dict:
        df = analyze_text(topn_input, "自动分词") 
        
        if not df.empty:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce').fillna(99999)
            
            # Top N 逻辑
            special_mask = df['cat'].isin(['term', 'proper'])
            general_mask = (df['cat'] == 'general') & (df['rank'] >= 2000) 
            valid_candidates = df[special_mask | general_mask].copy()
            sorted_df = valid_candidates.sort_values(by='rank', ascending=True)
            
            top_df = sorted_df.head(top_n)
            
            all_ids = set(df.index)
            top_ids = set(top_df.index)
            rest_ids = all_ids - top_ids
            rest_df = df.loc[list(rest_ids)].sort_values(by='rank')
            
            st.divider()
            col_win, col_rest = st.columns(2)
            
            # === 左栏：精选词汇 ===
            with col_win:
                st.success(f"🔥 精选 Top {len(top_df)} (由易到难)")
                if not top_df.empty:
                    words = top_df['word'].tolist()
                    with st.expander("👁️ 查看单词列表", expanded=True):
                        st.code("\n".join(words), language='text')
                    
                    st.markdown("**🤖 AI 制卡指令**")
                    has_term = any('(' in w for w in words)
                    
                    p_csv = generate_ai_prompt(words, 'csv', is_term_list=has_term)
                    p_txt = generate_ai_prompt(words, 'txt', is_term_list=has_term)
                    
                    # === 优化：使用 Tabs 展示 AI 指令 ===
                    t_csv, t_txt = st.tabs(["📋 CSV 指令", "📝 TXT 指令"])
                    with t_csv: st.code(p_csv, language='markdown')
                    with t_txt: st.code(p_txt, language='markdown')
                else:
                    st.warning("没有符合条件的单词 (可能都太简单了)")

            # === 右栏：剩余词汇 ===
            with col_rest:
                st.secondary_header(f"💤 剩余 {len(rest_df)} 个")
                if not rest_df.empty:
                    words_rest = rest_df['word'].tolist()
                    with st.expander("👁️ 查看剩余列表", expanded=False):
                        st.code("\n".join(words_rest), language='text')
                    
                    st.markdown("**🤖 AI 制卡指令**")
                    has_term_rest = any('(' in w for w in words_rest)
                    
                    p_csv_r = generate_ai_prompt(words_rest, 'csv', is_term_list=has_term_rest)
                    p_txt_r = generate_ai_prompt(words_rest, 'txt', is_term_list=has_term_rest)
                    
                    # === 优化：使用 Tabs 展示 AI 指令 ===
                    rt_csv, rt_txt = st.tabs(["📋 CSV 指令", "📝 TXT 指令"])
                    with rt_csv: st.code(p_csv_r, language='markdown')
                    with rt_txt: st.code(p_txt_r, language='markdown')