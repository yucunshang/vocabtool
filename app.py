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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 【核心】内置专业术语库 (带学科标签)
# ==========================================
BUILTIN_TECHNICAL_TERMS = {
    # === Computer Science (CS/AI) ===
    "algorithm": "CS", "recursion": "CS", "latency": "CS", "throughput": "CS", "bandwidth": "CS",
    "api": "CS", "json": "CS", "backend": "CS", "frontend": "CS", "fullstack": "CS",
    "neural": "AI", "transformer": "AI", "embedding": "AI", "inference": "AI", "perceptron": "AI",
    "python": "CS", "java": "CS", "docker": "CS", "kubernetes": "CS", "linux": "CS",
    "database": "CS", "cache": "CS", "compiler": "CS", "framework": "CS", "protocol": "CS",
    "encryption": "CS", "hash": "CS", "authentication": "CS", "authorization": "CS", "cryptography": "CS",
    "kernel": "CS", "shell": "CS", "terminal": "CS", "repository": "CS", "commit": "CS",
    "deployment": "CS", "iteration": "CS", "agile": "CS", "polymorphism": "CS", "encapsulation": "CS",
    "inheritance": "CS", "instantiation": "CS", "middleware": "CS", "scalability": "CS", "redundancy": "CS",
    "virtualization": "CS", "hypervisor": "CS", "container": "CS", "microservice": "CS", "serverless": "CS",
    "debugging": "CS", "syntax": "CS", "variable": "CS", "boolean": "CS", "integer": "CS",
    "array": "CS", "pointer": "CS", "reference": "CS", "memory": "CS", "cpu": "CS",
    "gpu": "CS", "binary": "CS", "hexadecimal": "CS", "bit": "CS", "byte": "CS",

    # === Mathematics (Math) ===
    "derivative": "Math", "integral": "Math", "limit": "Math", "calculus": "Math", "differential": "Math",
    "matrix": "Math", "vector": "Math", "scalar": "Math", "tensor": "Math", "determinant": "Math",
    "theorem": "Math", "axiom": "Math", "hypothesis": "Math", "lemma": "Math", "corollary": "Math",
    "variance": "Math", "deviation": "Math", "correlation": "Math", "regression": "Math", "covariance": "Math",
    "polynomial": "Math", "quadratic": "Math", "logarithm": "Math", "exponential": "Math", "arithmetic": "Math",
    "fraction": "Math", "decimal": "Math", "coefficient": "Math", "denominator": "Math", "numerator": "Math",
    "probability": "Math", "statistics": "Math", "permutation": "Math", "combination": "Math", "factorial": "Math",
    "geometry": "Math", "algebra": "Math", "trigonometry": "Math", "hypotenuse": "Math", "perimeter": "Math",
    "circumference": "Math", "radius": "Math", "diameter": "Math", "tangent": "Math", "cosine": "Math",
    "sine": "Math", "asymptote": "Math", "parabola": "Math", "ellipse": "Math", "hyperbola": "Math",

    # === Physics (Phys) ===
    "velocity": "Phys", "acceleration": "Phys", "momentum": "Phys", "inertia": "Phys", "trajectory": "Phys",
    "thermodynamics": "Phys", "entropy": "Phys", "enthalpy": "Phys", "kinetic": "Phys", "static": "Phys",
    "quantum": "Phys", "resonance": "Phys", "photon": "Phys", "electron": "Phys", "positron": "Phys",
    "proton": "Phys", "neutron": "Phys", "nucleus": "Phys", "atom": "Phys", "molecule": "Phys",
    "relativity": "Phys", "magnetism": "Phys", "voltage": "Phys", "amperage": "Phys", "capacitance": "Phys",
    "resistance": "Phys", "optics": "Phys", "refraction": "Phys", "reflection": "Phys", "diffraction": "Phys",
    "fission": "Phys", "fusion": "Phys", "radioactivity": "Phys", "isotope": "Phys", "half-life": "Phys",
    "gravity": "Phys", "friction": "Phys", "torque": "Phys", "oscillation": "Phys", "frequency": "Phys",
    "wavelength": "Phys", "amplitude": "Phys", "doppler": "Phys", "spectrum": "Phys", "vacuum": "Phys",

    # === Chemistry (Chem) ===
    "compound": "Chem", "solvent": "Chem", "solute": "Chem", "concentration": "Chem", "precipitate": "Chem",
    "alkali": "Chem", "catalyst": "Chem", "enzyme": "Chem", "substrate": "Chem", "reagent": "Chem",
    "covalent": "Chem", "ionic": "Chem", "oxidation": "Chem", "reduction": "Chem", "electrolysis": "Chem",
    "anion": "Chem", "cation": "Chem", "polymer": "Chem", "monomer": "Chem", "molecule": "Chem",
    "organic": "Chem", "inorganic": "Chem", "distillation": "Chem", "titration": "Chem", "filtration": "Chem",
    "hydrocarbon": "Chem", "carbohydrate": "Chem", "protein": "Chem", "lipid": "Chem", "amino": "Chem",
    "stoichiometry": "Chem", "equilibrium": "Chem", "thermodynamics": "Chem", "kinetics": "Chem", "activation": "Chem",
    "periodic": "Chem", "element": "Chem", "halogen": "Chem", "noble": "Chem", "metal": "Chem",
    
    # === Biology/Medicine (Bio/Med) ===
    "tissue": "Bio", "organ": "Bio", "organism": "Bio", "species": "Bio", "genus": "Bio",
    "mitochondria": "Bio", "ribosome": "Bio", "membrane": "Bio", "cytoplasm": "Bio", "chloroplast": "Bio",
    "dna": "Bio", "rna": "Bio", "chromosome": "Bio", "genome": "Bio", "allele": "Bio",
    "metabolism": "Bio", "photosynthesis": "Bio", "respiration": "Bio", "fermentation": "Bio", "homeostasis": "Bio",
    "evolution": "Bio", "mutation": "Bio", "selection": "Bio", "adaptation": "Bio", "symbiosis": "Bio",
    "pathogen": "Med", "antibody": "Med", "antigen": "Med", "vaccine": "Med", "immunity": "Med",
    "inflammation": "Med", "diagnosis": "Med", "prognosis": "Med", "symptom": "Med", "syndrome": "Med",
    "anatomy": "Med", "physiology": "Med", "pathology": "Med", "pharmacology": "Med", "toxicology": "Med",
    "cardiovascular": "Med", "respiratory": "Med", "neurology": "Med", "oncology": "Med", "pediatrics": "Med",

    # === Business/Finance (Biz) ===
    "revenue": "Biz", "margin": "Biz", "liability": "Biz", "equity": "Biz", "dividend": "Biz",
    "audit": "Biz", "fiscal": "Biz", "budget": "Biz", "forecast": "Biz", "overhead": "Biz",
    "stakeholder": "Biz", "shareholder": "Biz", "acquisition": "Biz", "ipo": "Biz", "merger": "Biz",
    "inflation": "Econ", "deflation": "Econ", "recession": "Econ", "gdp": "Econ", "macroeconomics": "Econ",
    "collateral": "Biz", "liquidity": "Biz", "bankruptcy": "Biz", "portfolio": "Biz", "diversification": "Biz",
    "amortization": "Biz", "depreciation": "Biz", "asset": "Biz", "capital": "Biz", "investment": "Biz",
    "arbitrage": "Biz", "derivative": "Biz", "hedge": "Biz", "leverage": "Biz", "valuation": "Biz",
    "entrepreneur": "Biz", "startup": "Biz", "venture": "Biz", "incubator": "Biz", "accelerator": "Biz",

    # === Law (Law) ===
    "plaintiff": "Law", "defendant": "Law", "verdict": "Law", "prosecutor": "Law", "juror": "Law",
    "appeal": "Law", "petition": "Law", "motion": "Law", "tort": "Law", "litigation": "Law",
    "felony": "Law", "misdemeanor": "Law", "affidavit": "Law", "subpoena": "Law", "warrant": "Law",
    "indictment": "Law", "arraignment": "Law", "acquittal": "Law", "conviction": "Law", "probation": "Law",
    "attorney": "Law", "jurisdiction": "Law", "arbitration": "Law", "mediation": "Law", "statute": "Law",
    "constitution": "Law", "amendment": "Law", "treaty": "Law", "contract": "Law", "clause": "Law",
    "liability": "Law", "negligence": "Law", "malpractice": "Law", "damages": "Law", "compensation": "Law",
    "intellectual": "Law", "copyright": "Law", "trademark": "Law", "patent": "Law", "infringement": "Law"
}

# ==========================================
# 3. 内置扩充词库 (Patch) & 专有名词
# ==========================================
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
# 4. 初始化 NLP
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

def get_word_info(raw_word):
    word_lower = raw_word.lower()
    word_clean = raw_word.strip()
    
    # 0. 检查内置专业术语 (返回 word + domain)
    if word_lower in BUILTIN_TECHNICAL_TERMS:
        domain = BUILTIN_TECHNICAL_TERMS[word_lower]
        return raw_word.strip(), f"term:{domain}"

    # 1. 检查歧义词
    if word_lower in AMBIGUOUS_WORDS:
        if word_clean[0].isupper(): return word_clean.title(), True
        else: return word_lower, False

    # 2. 检查纯专有名词库
    if word_lower in PROPER_NOUNS_DB:
        return PROPER_NOUNS_DB[word_lower], True
        
    return word_lower, False

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
# 5. 词库加载
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
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 6. AI 指令生成器 (核心优化)
# ==========================================
def generate_ai_prompt(word_list, output_format, is_term_list=False):
    words_str = ", ".join(word_list)
    if output_format == 'csv':
        format_req = "CSV Code Block (后缀名 .csv)"
        format_desc = "请直接输出标准 CSV 代码块。"
    else:
        format_req = "TXT Code Block (后缀名 .txt)"
        format_desc = "请输出纯文本 TXT 代码块。"

    # === 关键优化：针对术语列表的 Prompt ===
    context_instruction = ""
    if is_term_list:
        context_instruction = "\n- 注意：这些单词是【带领域标签的专业术语 (e.g. word (Domain))】。**英文释义**请务必根据括号内的领域（如 Math, CS）提供该领域的精确释义。**中文解析**部分请优先拆解【词源、词根、词缀】以辅助记忆；只有当英文释义非常晦涩难懂时，才补充中文领域解释，否则请聚焦于词源分析。"

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
# 7. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (Etymology)")

# === 高级设置折叠区 ===
with st.expander("⚙️ 词库与术语统计 (点击展开)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if vocab_dict:
            st.metric("📊 本地词库", f"{len(vocab_dict):,} 词")
        else:
            st.error("⚠️ 本地词库未加载")
    with c2:
        st.metric("🟣 内置术语库", f"{len(BUILTIN_TECHNICAL_TERMS)} 词", help="涵盖 CS, Math, Phys, Chem, Bio, Biz, Law")

# === 顶部功能切换 ===
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
    st.caption("功能：根据词频筛选生词，并生成 AI 制卡指令")
    col_level1, col_level2, col_space = st.columns([1, 1, 2])
    with col_level1:
        current_level = st.number_input("当前水平 (词频)", 0, 30000, 9000, 500)
    with col_level2:
        target_level = st.number_input("目标水平 (词频)", 0, 30000, 15000, 500)
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词", "按行处理"), horizontal=True)
        # 示例词现在展示了不同领域的
        grade_input = st.text_area("input_box", height=400, placeholder="variable\nlatency\ncell\ntort", label_visibility="collapsed")
        btn_grade = st.button("开始分级", type="primary", use_container_width=True)

    with g_col2:
        if not vocab_dict:
            st.error("❌ 词库未加载")
        elif btn_grade and grade_input:
            
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
                    
                    # === 获取信息 ===
                    display_word, info_type = get_word_info(item_cleaned)
                    
                    # 默认值
                    cat = "known" # 默认
                    rank = vocab_dict.get(item_lower, 99999)
                    
                    # 术语处理
                    if isinstance(info_type, str) and info_type.startswith("term:"):
                        cat = "term"
                        # 提取 domain
                        domain_str = info_type.split(":")[1]
                        display_word = f"{display_word} ({domain_str})"
                    
                    # 专有名词处理
                    elif info_type == True: 
                        cat = "proper"
                    
                    # 普通词处理
                    else:
                        if rank <= current_level: cat = "known"
                        elif rank <= target_level: cat = "target"
                        else: cat = "beyond"
                    
                    seen.add(item_lower)
                    unique_items.append({"word": display_word, "rank": rank, "cat": cat})
            
            df = pd.DataFrame(unique_items)
            if not df.empty:
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
                        count = len(words)
                        with st.expander(f"👁️ 查看/复制 {label} 列表 (共 {count} 个)", expanded=False):
                            st.code("\n".join(words), language='text')
                            st.caption("👆 复制单词列表")
                        
                        st.markdown(f"**🤖 AI 制卡指令 ({label})**")
                        # 传入 is_term
                        prompt_csv = generate_ai_prompt(words, 'csv', is_term_list=is_term)
                        prompt_txt = generate_ai_prompt(words, 'txt', is_term_list=is_term)
                        
                        ac1, ac2 = st.columns(2)
                        with ac1:
                            st.markdown("##### 📋 CSV 版")
                            st.code(prompt_csv, language='markdown')
                        with ac2:
                            st.markdown("##### 📝 TXT 版")
                            st.code(prompt_txt, language='markdown')

                with t_term: show("term", "专业术语", is_term=True)
                with t_target: show("target", "重点词")
                with t_proper: show("proper", "专有名词")
                with t_beyond: show("beyond", "超纲词")
                with t_known: show("known", "熟词")
            else:
                st.warning("无有效单词")