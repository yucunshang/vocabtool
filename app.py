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
    /* 侧边栏样式优化 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 内置数据 (Technical Terms & Proper Nouns)
# ==========================================
BUILTIN_TECHNICAL_TERMS = {
    # 用户指定补充
    "metal": "Chem", "motion": "Law", "gravity": "Phys", "molecule": "Chem",
    "vacuum": "Phys", "electron": "Phys", "quantum": "Phys", "velocity": "Phys",
    "friction": "Phys", "catalyst": "Chem", "equilibrium": "Chem",
    
    # CS
    "algorithm": "CS", "recursion": "CS", "latency": "CS", "throughput": "CS", "bandwidth": "CS",
    "api": "CS", "json": "CS", "backend": "CS", "frontend": "CS", "fullstack": "CS",
    "neural": "AI", "transformer": "AI", "embedding": "AI", "inference": "AI",
    "python": "CS", "java": "CS", "docker": "CS", "kubernetes": "CS", "linux": "CS",
    "database": "CS", "cache": "CS", "compiler": "CS", "framework": "CS",
    "encryption": "CS", "hash": "CS", "authentication": "CS", "authorization": "CS",
    "kernel": "CS", "shell": "CS", "terminal": "CS", "repository": "CS", "commit": "CS",
    "deployment": "CS", "iteration": "CS", "agile": "CS", "polymorphism": "CS",
    "inheritance": "CS", "instantiation": "CS", "middleware": "CS", "scalability": "CS",

    # Math
    "derivative": "Math", "integral": "Math", "limit": "Math", "calculus": "Math",
    "matrix": "Math", "vector": "Math", "scalar": "Math", "tensor": "Math",
    "theorem": "Math", "axiom": "Math", "hypothesis": "Math", "lemma": "Math",
    "variance": "Math", "deviation": "Math", "correlation": "Math", "regression": "Math",
    "polynomial": "Math", "quadratic": "Math", "logarithm": "Math", "exponential": "Math",
    "integer": "Math", "fraction": "Math", "decimal": "Math", "coefficient": "Math",
    "probability": "Math", "statistics": "Math", "permutation": "Math", "combination": "Math",

    # Phys
    "acceleration": "Phys", "momentum": "Phys", "inertia": "Phys",
    "thermodynamics": "Phys", "entropy": "Phys", "enthalpy": "Phys", "kinetic": "Phys",
    "resonance": "Phys", "photon": "Phys", "positron": "Phys",
    "proton": "Phys", "neutron": "Phys", "nucleus": "Phys", "atom": "Phys",
    "relativity": "Phys", "magnetism": "Phys", "voltage": "Phys", "amperage": "Phys",
    "resistance": "Phys", "optics": "Phys", "refraction": "Phys", "reflection": "Phys",

    # Chem
    "compound": "Chem", "solvent": "Chem", "solute": "Chem", "concentration": "Chem",
    "alkali": "Chem", "enzyme": "Chem", "substrate": "Chem", "reagent": "Chem",
    "covalent": "Chem", "ionic": "Chem", "oxidation": "Chem", "reduction": "Chem",
    "isotope": "Chem", "anion": "Chem", "cation": "Chem", "polymer": "Chem",
    "monomer": "Chem", "organic": "Chem", "inorganic": "Chem", "distillation": "Chem",
    "titration": "Chem", "filtration": "Chem", "hydrocarbon": "Chem",

    # Bio
    "tissue": "Bio", "organ": "Bio", "organism": "Bio",
    "mitochondria": "Bio", "ribosome": "Bio", "membrane": "Bio", "cytoplasm": "Bio",
    "dna": "Bio", "rna": "Bio", "chromosome": "Bio", "genome": "Bio",
    "protein": "Bio", "lipid": "Bio", "carbohydrate": "Bio", "vitamin": "Bio",
    "photosynthesis": "Bio", "metabolism": "Bio", "evolution": "Bio", "mutation": "Bio",
    "pathogen": "Med", "antibody": "Med", "antigen": "Med", "vaccine": "Med",
    "inflammation": "Med", "diagnosis": "Med", "prognosis": "Med", "symptom": "Med",
    "anatomy": "Med", "physiology": "Med", "pathology": "Med", "pharmacology": "Med",

    # Biz
    "revenue": "Biz", "margin": "Biz", "liability": "Biz", "equity": "Biz", "dividend": "Biz",
    "audit": "Biz", "fiscal": "Biz", "budget": "Biz", "forecast": "Biz",
    "stakeholder": "Biz", "shareholder": "Biz", "acquisition": "Biz", "ipo": "Biz",
    "inflation": "Econ", "deflation": "Econ", "recession": "Econ", "gdp": "Econ",
    "collateral": "Biz", "liquidity": "Biz", "bankruptcy": "Biz", "portfolio": "Biz",

    # Law
    "plaintiff": "Law", "defendant": "Law", "verdict": "Law", "prosecutor": "Law",
    "appeal": "Law", "petition": "Law", "motion": "Law", "tort": "Law",
    "felony": "Law", "misdemeanor": "Law", "affidavit": "Law", "subpoena": "Law",
    "indictment": "Law", "litigation": "Law", "attorney": "Law", "jurisdiction": "Law",
    "arbitration": "Law", "statute": "Law", "constitution": "Law"
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
    
    # 内置补丁
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
    
    for word, rank in BUILTIN_PATCH_VOCAB.items():
        if word not in vocab:
            vocab[word] = rank
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
        context_instruction = "\n- 注意：这些单词是【带领域标签的专业术语 (e.g. word (Domain))】。**英文释义**请务必根据括号内的领域（如 Math, CS）提供该领域的精确释义。**中文解析**部分请优先拆解【词源、词根、词缀】以辅助记忆；只有当英文释义非常晦涩难懂时，才补充中文领域解释，否则请聚焦于词源分析。"

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
# 6. 辅助函数：智能精选 (Top N Selector)
# ==========================================
def get_top_n_words(df, n_count, current_level):
    """
    筛选逻辑：
    1. 排除 Rank < 2000 的词 (太简单)
    2. 排除 Rank > 20000 的词 (太生僻, 除非是术语)
    3. 优先选择 Rank 在 current_level 附近的词 (学习区)
    4. 按 Rank 由易到难排序
    """
    # 过滤掉非普通词 (术语和专有名词单算，这里只筛选普通词)
    candidates = df[df['cat'].isin(['target', 'beyond', 'known'])].copy()
    
    # 强制类型转换，防止 rank 是字符串
    candidates['rank'] = pd.to_numeric(candidates['rank'], errors='coerce').fillna(99999)
    
    # 核心过滤：只看 2000 ~ 20000 之间的词 (黄金区间)
    mask = (candidates['rank'] >= 2000) & (candidates['rank'] <= 22000)
    golden_candidates = candidates[mask]
    
    # 如果黄金区间不够数，就放宽限制
    if len(golden_candidates) < n_count:
        final_list = candidates.sort_values(by='rank').head(n_count)
    else:
        # 在黄金区间里，按 rank 排序
        final_list = golden_candidates.sort_values(by='rank').head(n_count)
        
    return final_list['word'].tolist()

# ==========================================
# 7. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (Smart Select)")

# === 侧边栏：智能精选入口 ===
with st.sidebar:
    st.header("🎯 智能精选 (Top N)")
    st.info("当文章太长、生词太多时，用这个功能筛选出“性价比最高”的词汇。")
    top_n_num = st.number_input("筛选数量", 10, 500, 50, 10)
    # 使用 session_state 来触发筛选
    if st.button("🎲 生成精选词单", type="primary"):
        st.session_state['trigger_top_n'] = True
    else:
        # 保持状态，除非重新分析
        if 'trigger_top_n' not in st.session_state:
            st.session_state['trigger_top_n'] = False

    st.divider()
    if vocab_dict:
        st.caption(f"📚 本地词库: {len(vocab_dict):,} 词")

# === 顶部 Tab ===
st.divider()
app_mode = st.radio("选择功能模式:", ["🛠️ 智能还原", "📊 单词分级 (AI 制卡)"], horizontal=True)

if "智能还原" in app_mode:
    c1, c2 = st.columns(2)
    with c1:
        raw_text = st.text_area("输入原始文章", height=400, placeholder="He was excited.")
        btn_restore = st.button("开始还原", type="primary")
    with c2:
        if btn_restore and raw_text:
            res = smart_lemmatize(raw_text)
            st.code(res, language='text')
            st.caption("👆 一键复制")
        elif not raw_text: st.info("👈 请输入文本")

else:
    # 分级模式
    col_level1, col_level2, col_space = st.columns([1, 1, 2])
    with col_level1:
        current_level = st.number_input("当前水平 (词频)", 0, 30000, 9000, 500)
    with col_level2:
        target_level = st.number_input("目标水平 (词频)", 0, 30000, 15000, 500)
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词", "按行处理"), horizontal=True)
        grade_input = st.text_area("input_box", height=400, placeholder="motion\nmetal\nenergy\nrevenue\nabacus\nabandon", label_visibility="collapsed")
        
        # 当点击“开始分级”时，重置 Top N 状态
        if st.button("开始分级", type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
            st.session_state['trigger_top_n'] = False # 重置筛选
        
    with g_col2:
        # 检查是否需要运行分析
        if st.session_state.get('run_analysis', False) and grade_input and vocab_dict:
            
            # --- 数据处理逻辑 ---
            raw_items = []
            if "按行" in input_mode:
                lines = grade_input.split('\n')
                for line in lines:
                    if line.strip(): raw_items.append(line.strip())
            else:
                raw_items = grade_input.split()
            
            seen = set()
            unique_items = [] 
            JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're'}
            
            with st.spinner("正在智能分析..."):
                for item in raw_items:
                    item_cleaned = item.strip()
                    item_lower = item_cleaned.lower()
                    
                    if item_lower in seen: continue
                    if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
                    if item_lower in JUNK_WORDS: continue
                    
                    # 1. 术语
                    if item_lower in BUILTIN_TECHNICAL_TERMS:
                        domain = BUILTIN_TECHNICAL_TERMS[item_lower]
                        unique_items.append({"word": f"{item_cleaned} ({domain})", "rank": 0, "cat": "term"})
                    
                    # 2. 专名
                    if item_lower in PROPER_NOUNS_DB:
                        unique_items.append({"word": PROPER_NOUNS_DB[item_lower], "rank": 0, "cat": "proper"})
                        
                    # 3. 普通词
                    rank = vocab_dict.get(item_lower, 99999)
                    if rank != 99999:
                        if rank <= current_level: cat = "known"
                        elif rank <= target_level: cat = "target"
                        else: cat = "beyond"
                        unique_items.append({"word": item_cleaned, "rank": rank, "cat": cat})
                    
                    seen.add(item_lower)
            
            # 保存到 session_state 以便复用
            df = pd.DataFrame(unique_items)
            st.session_state['df_result'] = df

        # --- 展示结果逻辑 ---
        if 'df_result' in st.session_state and not st.session_state['df_result'].empty:
            df = st.session_state['df_result']
            
            # 如果用户点击了侧边栏的“生成精选词单”
            if st.session_state.get('trigger_top_n', False):
                st.success(f"🎯 已为您精选 Top {top_n_num} 个最值得学习的单词 (Rank 2000+)")
                top_words = get_top_n_words(df, top_n_num, current_level)
                
                if top_words:
                    with st.expander(f"🔥 精选词单 ({len(top_words)} 个)", expanded=True):
                        st.code("\n".join(top_words), language='text')
                        st.markdown("**🤖 AI 制卡指令 (精选版)**")
                        p_csv = generate_ai_prompt(top_words, 'csv')
                        p_txt = generate_ai_prompt(top_words, 'txt')
                        c1, c2 = st.columns(2)
                        with c1: st.code(p_csv, language='markdown')
                        with c2: st.code(p_txt, language='markdown')
                else:
                    st.warning("词汇太简单或太少，无法筛选。")
                st.divider()

            # 常规展示 (Tabs)
            df = df.sort_values(by='rank', ascending=True)
            t_term, t_target, t_proper, t_beyond, t_known = st.tabs([
                f"🟣 专业术语 ({len(df[df['cat']=='term'])})",
                f"🟡 重点 ({len(df[df['cat']=='target'])})", 
                f"🔵 专有名词 ({len(df[df['cat']=='proper'])})", 
                f"🔴 超纲 ({len(df[df['cat']=='beyond'])})", 
                f"🟢 已掌握 ({len(df[df['cat']=='known'])})"
            ])
            
            def show(cat_name, label, is_term=False):
                sub = df[df['cat'] == cat_name]
                if sub.empty: 
                    st.info("无")
                else:
                    words = sub['word'].tolist()
                    with st.expander(f"👁️ 查看 {label} ({len(words)})", expanded=False):
                        st.code("\n".join(words), language='text')
                    
                    st.markdown(f"**🤖 AI 指令 ({label})**")
                    prompt_csv = generate_ai_prompt(words, 'csv', is_term_list=is_term)
                    prompt_txt = generate_ai_prompt(words, 'txt', is_term_list=is_term)
                    c1, c2 = st.columns(2)
                    with c1: st.code(prompt_csv, language='markdown')
                    with c2: st.code(prompt_txt, language='markdown')

            with t_term: show("term", "专业术语", is_term=True)
            with t_target: show("target", "重点词")
            with t_proper: show("proper", "专有名词")
            with t_beyond: show("beyond", "超纲词")
            with t_known: show("known", "熟词")