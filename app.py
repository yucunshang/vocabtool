import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import io

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
    [data-testid="stSidebar"] { background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 内置扩充词库 (Hardcoded Patch)
# ==========================================

# A. 专有名词库 (White List) - 扩容版
PROPER_NOUNS_DB = {
    # 国家/地区
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
    
    # 城市
    "london": "London", "paris": "Paris", "tokyo": "Tokyo", "beijing": "Beijing",
    "shanghai": "Shanghai", "hong kong": "Hong Kong", "sydney": "Sydney", 
    "melbourne": "Melbourne", "berlin": "Berlin", "rome": "Rome", "madrid": "Madrid",
    "new york": "New York", "los angeles": "Los Angeles", "san francisco": "San Francisco",
    "chicago": "Chicago", "seattle": "Seattle", "boston": "Boston", "houston": "Houston",
    "moscow": "Moscow", "cairo": "Cairo", "dubai": "Dubai", "mumbai": "Mumbai",
    
    # 洲/洋
    "africa": "Africa", "asia": "Asia", "europe": "Europe", "antarctica": "Antarctica",
    "north america": "North America", "south america": "South America",
    "pacific": "Pacific", "atlantic": "Atlantic", "indian ocean": "Indian Ocean",
    
    # 时间/节日
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
    "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday",
    "january": "January", "february": "February", "march": "March", "april": "April", 
    "may": "May", "june": "June", "july": "July", "august": "August", 
    "september": "September", "october": "October", "november": "November", "december": "December",
    "christmas": "Christmas", "easter": "Easter", "thanksgiving": "Thanksgiving", "halloween": "Halloween",
    
    # 科技/品牌/机构
    "google": "Google", "apple": "Apple", "microsoft": "Microsoft", "tesla": "Tesla",
    "amazon": "Amazon", "facebook": "Facebook", "twitter": "Twitter", "youtube": "YouTube", "instagram": "Instagram",
    "tiktok": "TikTok", "netflix": "Netflix", "spotify": "Spotify", "zoom": "Zoom",
    "nasa": "NASA", "fbi": "FBI", "cia": "CIA", "un": "UN", "eu": "EU", "nato": "NATO", "wto": "WTO", "who": "WHO",
    "iphone": "iPhone", "ipad": "iPad", "mac": "Mac", "windows": "Windows", "android": "Android",
    "wifi": "Wi-Fi", "internet": "Internet", "bluetooth": "Bluetooth",
    
    # 常见称谓/学位
    "mr": "Mr.", "mrs": "Mrs.", "ms": "Ms.", "dr": "Dr.", "prof": "Prof.",
    "phd": "PhD", "mba": "MBA", "ceo": "CEO", "cfo": "CFO", "cto": "CTO", "vip": "VIP"
}

# B. 现代 & 学术 补丁词库 (Built-in Patch)
# 这些词通常在老旧 CSV 中缺失，或者排名不合理。我们手动强制注入。
# 设定 rank = 5000~8000 (属于“进阶但非生僻”)
BUILTIN_PATCH_VOCAB = {
    # 现代科技/互联网 (Modern Tech)
    "online": 2000, "website": 2500, "app": 3000, "user": 1500, "data": 1000,
    "software": 3000, "hardware": 4000, "network": 2500, "server": 3500,
    "cloud": 3000, "algorithm": 6000, "database": 5000, "interface": 5000,
    "digital": 3000, "virtual": 4000, "smart": 2000, "mobile": 2500,
    "email": 2000, "text": 1000, "chat": 2000, "video": 1500, "audio": 3000,
    "link": 2000, "click": 2000, "search": 1500, "share": 1500, "post": 1500,
    
    # 常见学术/商务 (Academic/Business)
    "analysis": 2500, "strategy": 2500, "method": 2000, "theory": 2500,
    "research": 1500, "evidence": 2000, "significant": 2000, "factor": 1500,
    "process": 1000, "system": 1000, "available": 1500, "similar": 1500,
    "specific": 2000, "issue": 1000, "policy": 1500, "community": 1500,
    "development": 1500, "economic": 2000, "global": 2500, "environment": 2000,
    "challenge": 2500, "opportunity": 2000, "solution": 2500, "management": 2500,
    
    # 容易被误判的“小词”
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

def load_custom_terms(uploaded_file):
    if uploaded_file is None: return set()
    terms = set()
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        for line in stringio:
            parts = line.replace(',', '\n').split('\n')
            for p in parts:
                clean_w = p.strip().lower()
                if clean_w: terms.add(clean_w)
    except: pass
    return terms

def get_word_info(raw_word, custom_terms_set):
    word_lower = raw_word.lower()
    word_clean = raw_word.strip()
    
    if word_lower in custom_terms_set:
        return raw_word.strip(), "term"
    if word_lower in AMBIGUOUS_WORDS:
        if word_clean[0].isupper(): return word_clean.title(), True
        else: return word_lower, False
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
# 4. 词库加载 (核心优化：CSV + 补丁合并)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    vocab = {}
    
    # 1. 加载本地 CSV (如果有)
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
            
            # 清洗
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
            
            # 去重：按 Rank 排序，保留 Rank 最小的那个
            df = df.sort_values(r_col, ascending=True)
            df = df.drop_duplicates(subset=[w_col], keep='first')
            
            vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except: pass
    
    # 2. 注入内置补丁 (Built-in Patch)
    # 逻辑：如果 CSV 里没有这个词，或者 CSV 里的 rank 太大(>20000)，用补丁覆盖
    for word, rank in BUILTIN_PATCH_VOCAB.items():
        if word not in vocab:
            vocab[word] = rank
        else:
            # 如果 CSV 里有，但排名极其靠后(比如错排到 60000)，我们把它拉回来
            if vocab[word] > 20000:
                vocab[word] = rank
                
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 5. AI 指令生成器
# ==========================================
def generate_ai_prompt(word_list, output_format, is_term_list=False):
    words_str = ", ".join(word_list)
    if output_format == 'csv':
        format_req = "CSV Code Block (后缀名 .csv)"
        format_desc = "请直接输出标准 CSV 代码块。"
    else:
        format_req = "TXT Code Block (后缀名 .txt)"
        format_desc = "请输出纯文本 TXT 代码块。"

    context_instruction = ""
    if is_term_list:
        context_instruction = "\n- 注意：这些单词是【专业术语 (Technical Terms)】，请提供其在特定专业领域（如科技、医学、法律）中的精确释义，而非通用含义。"

    prompt = f"""
请扮演一位专业的 Anki 制卡专家。这是我整理的单词列表{context_instruction}，请严格按照以下【终极制卡标准】为我生成导入文件。

1. 核心原则：原子性 (Atomicity)
- 含义拆分：若单词有多个不同含义，拆分为多条数据。
- 严禁堆砌：每张卡片只承载一个特定语境下的含义。

2. 卡片正面 (Column 1: Front)
- 内容：提供自然的短语或搭配 (Phrase/Collocation)，而非单个孤立单词。
- 样式：纯文本，不加粗。

3. 卡片背面 (Column 2: Back)
- 格式：HTML 排版，包含三部分，必须使用 <br><br> 分隔。
- 结构：英文释义<br><br><em>斜体例句</em><br><br>【词根词缀/术语解析】中文解析

4. 输出格式标准 ({format_req})
- {format_desc}
- 关键格式：使用英文逗号 (,) 分隔，且每个字段内容必须用英文双引号 ("...") 包裹 (防止 HTML 内容冲突)。
- 示例： "Front Content","Back Content"

待处理单词：
{words_str}
"""
    return prompt

# ==========================================
# 6. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (Extended)")

# === 侧边栏 ===
with st.sidebar:
    st.header("⚙️ 全局设置")
    
    # 显示词库状态
    if vocab_dict:
        total_vocab = len(vocab_dict)
        st.metric("📊 实际词库容量", f"{total_vocab:,}", delta="已加载补丁")
    else:
        st.error("⚠️ 未加载本地 CSV")

    st.subheader("1. 词汇量分级")
    # 提升上限到 30000
    current_level = st.number_input("当前水平 (Current)", 0, 30000, 9000, 500)
    target_level = st.number_input("目标水平 (Target)", 0, 30000, 15000, 500)
    
    st.divider()
    
    st.subheader("2. 本地术语库 (可选)")
    uploaded_terms = st.file_uploader("上传 CSV/TXT 文件", type=['csv', 'txt'])
    custom_terms_set = load_custom_terms(uploaded_terms)
    if custom_terms_set: st.success(f"已加载 {len(custom_terms_set)} 个自定义术语")

# === 主功能区 ===
app_mode = st.radio("选择功能模式:", ["🛠️ 智能还原", "📊 单词分级 (AI 制卡)"], horizontal=True)
st.divider()

if "智能还原" in app_mode:
    c1, c2 = st.columns(2)
    with c1:
        raw_text = st.text_area("输入原始文章", height=400, placeholder="He was excited.")
        btn_restore = st.button("开始还原", type="primary")
    with c2:
        if btn_restore and raw_text:
            res = smart_lemmatize(raw_text)
            st.code(res, language='text')
            st.caption("👆 点击右上角图标，一键复制还原后的文本")
        elif not raw_text: st.info("👈 请输入文本")

else:
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词", "按行处理"), horizontal=True)
        # 增加几个测试词 (algorithm, online 是补丁词)
        grade_input = st.text_area("input_box", height=400, placeholder="algorithm\nonline\nChina\nshove", label_visibility="collapsed")
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
                    
                    display_word, info_type = get_word_info(item_cleaned, custom_terms_set)
                    
                    # 查词：此时 vocab_dict 已经包含了补丁词
                    rank = vocab_dict.get(item_lower, 99999)
                    
                    if info_type == "term": cat = "term"
                    elif info_type == True: cat = "proper"
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