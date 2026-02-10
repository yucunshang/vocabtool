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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 内置巨型专有名词库 (大小写映射表)
# ==========================================
# key = 全小写, value = 正确显示格式
PROPER_NOUNS_DB = {
    # --- 国家 & 地区 ---
    "china": "China", "usa": "USA", "uk": "UK", "america": "America", "england": "England",
    "japan": "Japan", "korea": "Korea", "france": "France", "germany": "Germany", "italy": "Italy",
    "spain": "Spain", "russia": "Russia", "india": "India", "brazil": "Brazil", "canada": "Canada",
    "australia": "Australia", "new zealand": "New Zealand", "mexico": "Mexico", "egypt": "Egypt",
    "singapore": "Singapore", "malaysia": "Malaysia", "thailand": "Thailand", "vietnam": "Vietnam",
    "switzerland": "Switzerland", "sweden": "Sweden", "norway": "Norway", "denmark": "Denmark",
    "finland": "Finland", "netherlands": "Netherlands", "belgium": "Belgium", "austria": "Austria",
    "greece": "Greece", "turkey": "Turkey", "israel": "Israel", "saudi arabia": "Saudi Arabia",
    "dubai": "Dubai", "africa": "Africa", "asia": "Asia", "europe": "Europe", "antarctica": "Antarctica",
    
    # --- 著名城市 ---
    "london": "London", "new york": "New York", "paris": "Paris", "tokyo": "Tokyo", "beijing": "Beijing",
    "shanghai": "Shanghai", "hong kong": "Hong Kong", "sydney": "Sydney", "melbourne": "Melbourne",
    "berlin": "Berlin", "rome": "Rome", "madrid": "Madrid", "moscow": "Moscow", "cairo": "Cairo",
    "los angeles": "Los Angeles", "san francisco": "San Francisco", "chicago": "Chicago", "seattle": "Seattle",
    "boston": "Boston", "washington": "Washington", "toronto": "Toronto", "vancouver": "Vancouver",
    
    # --- 时间 (星期/月份) ---
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
    "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday",
    "january": "January", "february": "February", "march": "March", "april": "April",
    "may": "May", "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November", "december": "December",
    
    # --- 常见英文名 (Top 50+) ---
    "james": "James", "john": "John", "robert": "Robert", "michael": "Michael", "william": "William",
    "david": "David", "richard": "Richard", "joseph": "Joseph", "thomas": "Thomas", "charles": "Charles",
    "mary": "Mary", "patricia": "Patricia", "jennifer": "Jennifer", "linda": "Linda", "elizabeth": "Elizabeth",
    "barbara": "Barbara", "susan": "Susan", "jessica": "Jessica", "sarah": "Sarah", "karen": "Karen",
    "trump": "Trump", "biden": "Biden", "obama": "Obama", "musk": "Musk", "jobs": "Jobs", "gates": "Gates",
    
    # --- 科技 & 品牌 ---
    "google": "Google", "apple": "Apple", "microsoft": "Microsoft", "amazon": "Amazon", "facebook": "Facebook",
    "tesla": "Tesla", "twitter": "Twitter", "instagram": "Instagram", "youtube": "YouTube", "tiktok": "TikTok",
    "iphone": "iPhone", "ipad": "iPad", "mac": "Mac", "windows": "Windows", "android": "Android",
    "nike": "Nike", "adidas": "Adidas", "coca-cola": "Coca-Cola", "pepsi": "Pepsi", "mcdonald's": "McDonald's",
    
    # --- 缩写 & 组织 ---
    "nasa": "NASA", "fbi": "FBI", "cia": "CIA", "un": "UN", "eu": "EU", "nato": "NATO",
    "ceo": "CEO", "cfo": "CFO", "cto": "CTO", "phd": "PhD", "mba": "MBA", "covid": "COVID"
}

# ==========================================
# 3. 初始化 NLP (本地下载修复)
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

def get_word_info(word):
    """
    核心判断逻辑：
    返回 (display_word, is_proper_noun)
    """
    word_lower = word.lower()
    
    # 1. 优先查内置大词库 (最准)
    if word_lower in PROPER_NOUNS_DB:
        return PROPER_NOUNS_DB[word_lower], True
        
    # 2. 如果词库没查到，用 NLTK 辅助判断 (针对生僻人名)
    try:
        test_word = word.title()
        tags = nltk.pos_tag([test_word])
        pos_tag = tags[0][1]
        if pos_tag.startswith('NNP'): # 专有名词
            return test_word, True
    except:
        pass
        
    # 3. 普通单词，强制小写
    return word_lower, False

# 还原引擎
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
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if not file_path: return None
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
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
    except: return None

vocab_dict = load_vocab()

# ==========================================
# 5. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (Proper Nouns)")

tab_lemma, tab_grade = st.tabs(["🛠️ 1. 智能还原", "📊 2. 单词分级"])

# --- Tab 1 ---
with tab_lemma:
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

# --- Tab 2 (包含专有名词分类) ---
with tab_grade:
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a: current_level = st.number_input("当前水平", 0, 20000, 9000, 500)
    with col_b: target_level = st.number_input("目标水平", 0, 20000, 15000, 500)
    st.divider()
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词", "按行处理"), horizontal=True)
        grade_input = st.text_area("input_box", height=400, placeholder="China\nanti\nJohn", label_visibility="collapsed")
        btn_grade = st.button("开始分级", type="primary", use_container_width=True)

    with g_col2:
        if not vocab_dict:
            st.error("❌ 词库未加载")
        elif btn_grade and grade_input:
            
            # 获取输入
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
            
            # 数据结构：(显示单词, rank, 类别)
            data = []
            
            with st.spinner("正在智能分析..."):
                for item in raw_items:
                    item_cleaned = item.strip()
                    item_lower = item_cleaned.lower()
                    
                    if item_lower in seen: continue
                    if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
                    if item_lower in JUNK_WORDS: continue
                    
                    # === 核心逻辑：获取显示格式 & 是否为专有名词 ===
                    display_word, is_proper = get_word_info(item_cleaned)
                    
                    # 查词频
                    rank = vocab_dict.get(item_lower, 99999)
                    
                    # 分类逻辑
                    if is_proper:
                        cat = "proper" # 新增：专有名词
                    else:
                        if rank <= current_level: cat = "known"
                        elif rank <= target_level: cat = "target"
                        else: cat = "beyond"
                    
                    seen.add(item_lower)
                    data.append({"word": display_word, "rank": rank, "cat": cat})
            
            # 生成 Tab
            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values(by='rank', ascending=True)
                
                # 定义 4 个 Tabs
                t1, t2, t3, t4 = st.tabs([
                    f"🟡 重点 ({len(df[df['cat']=='target'])})", 
                    f"🔵 专有名词 ({len(df[df['cat']=='proper'])})", 
                    f"🔴 超纲 ({len(df[df['cat']=='beyond'])})", 
                    f"🟢 已掌握 ({len(df[df['cat']=='known'])})"
                ])
                
                def show(cat_name):
                    sub = df[df['cat'] == cat_name]
                    if sub.empty: 
                        st.info("无")
                    else:
                        txt = "\n".join(sub['word'].tolist())
                        st.code(txt, language='text')
                        st.caption("👆 一键复制")

                with t1: show("target")
                with t2: show("proper") # 新增的专有名词 Tab
                with t3: show("beyond")
                with t4: show("known")
            else:
                st.warning("无有效单词")