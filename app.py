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
    
    /* 强调用途提示框 */
    .stAlert {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 内置词库 (地名/时间/品牌白名单)
# ==========================================
PROPER_NOUNS_DB = {
    "usa": "USA", "uk": "UK", "america": "America", "england": "England",
    "japan": "Japan", "korea": "Korea", "france": "France", "germany": "Germany", "italy": "Italy",
    "spain": "Spain", "russia": "Russia", "india": "India", "brazil": "Brazil", "canada": "Canada",
    "australia": "Australia", "mexico": "Mexico", "egypt": "Egypt", "china": "China",
    "switzerland": "Switzerland", "sweden": "Sweden", "norway": "Norway",
    "london": "London", "paris": "Paris", "tokyo": "Tokyo", "beijing": "Beijing",
    "shanghai": "Shanghai", "hong kong": "Hong Kong", "sydney": "Sydney", 
    "melbourne": "Melbourne", "berlin": "Berlin", "rome": "Rome",
    "new york": "New York", "los angeles": "Los Angeles", "san francisco": "San Francisco",
    "chicago": "Chicago", "seattle": "Seattle", "boston": "Boston",
    "moscow": "Moscow", "cairo": "Cairo", "dubai": "Dubai",
    "africa": "Africa", "asia": "Asia", "europe": "Europe", "antarctica": "Antarctica",
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
    "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday",
    "january": "January", "february": "February", "april": "April", 
    "june": "June", "july": "July", "september": "September", 
    "october": "October", "november": "November", "december": "December",
    "google": "Google", "apple": "Apple", "microsoft": "Microsoft", "tesla": "Tesla",
    "amazon": "Amazon", "facebook": "Facebook", "twitter": "Twitter", "youtube": "YouTube",
    "nasa": "NASA", "fbi": "FBI", "cia": "CIA", "un": "UN", "eu": "EU", "nato": "NATO",
    "iphone": "iPhone", "ipad": "iPad", "wifi": "Wi-Fi", "internet": "Internet"
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

def get_word_info(raw_word):
    word_lower = raw_word.lower()
    word_clean = raw_word.strip()
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
# 5. 核心：AI 指令生成器 (优化版)
# ==========================================
def generate_ai_prompt(word_list):
    """
    生成符合用户“终极目标”的 Prompt
    """
    words_str = ", ".join(word_list)
    
    prompt = f"""
请扮演一位专业的 Anki 制卡专家。这是我整理的单词列表，请严格按照以下【终极制卡标准】为我生成导入内容。

1. 核心原则：原子性 (Atomicity)
- 含义拆分：若单词有多个不同含义，拆分为多条数据。
- 严禁堆砌：每张卡片只承载一个特定语境下的含义。

2. 卡片正面 (Column 1: Front)
- 内容：提供自然的短语或搭配 (Phrase/Collocation)，而非单个孤立单词。
- 样式：纯文本，不加粗。

3. 卡片背面 (Column 2: Back)
- 格式：HTML 排版，包含三部分，必须使用 <br><br> 分隔。
- 结构：英文释义<br><br><em>斜体例句</em><br><br>【词根词缀】中文解析

4. 输出格式标准 (CSV Code Block)
- 请直接输出 CSV 代码块。
- 英文逗号分隔，每个字段用双引号包裹。

待处理单词：
{words_str}
"""
    return prompt

# ==========================================
# 6. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (AI Command)")

tab_lemma, tab_grade = st.tabs(["🛠️ 1. 智能还原", "📊 2. 单词分级 (含指令)"])

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
            st.caption("👆 点击右上角图标，一键复制还原后的文本")
        elif not raw_text: st.info("👈 请输入文本")

# --- Tab 2 ---
with tab_grade:
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a: current_level = st.number_input("当前水平", 0, 20000, 9000, 500)
    with col_b: target_level = st.number_input("目标水平", 0, 20000, 15000, 500)
    st.divider()
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词", "按行处理"), horizontal=True)
        grade_input = st.text_area("input_box", height=400, placeholder="China\nParis\nshove\nunhinge", label_visibility="collapsed")
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
                    
                    display_word, is_proper = get_word_info(item_cleaned)
                    rank = vocab_dict.get(item_lower, 99999)
                    
                    if is_proper: cat = "proper"
                    else:
                        if rank <= current_level: cat = "known"
                        elif rank <= target_level: cat = "target"
                        else: cat = "beyond"
                    
                    seen.add(item_lower)
                    unique_items.append({"word": display_word, "rank": rank, "cat": cat})
            
            df = pd.DataFrame(unique_items)
            if not df.empty:
                df = df.sort_values(by='rank', ascending=True)
                
                t1, t2, t3, t4 = st.tabs([
                    f"🟡 重点 ({len(df[df['cat']=='target'])})", 
                    f"🔵 专有名词 ({len(df[df['cat']=='proper'])})", 
                    f"🔴 超纲 ({len(df[df['cat']=='beyond'])})", 
                    f"🟢 已掌握 ({len(df[df['cat']=='known'])})"
                ])
                
                def show(cat_name, label):
                    sub = df[df['cat'] == cat_name]
                    if sub.empty: 
                        st.info("无")
                    else:
                        # 1. 单词列表 (一键复制)
                        st.markdown(f"**1. {label} 列表**")
                        words = sub['word'].tolist()
                        st.code("\n".join(words), language='text')
                        
                        # 2. AI 指令 (核心优化点)
                        st.divider()
                        st.markdown(f"**2. AI 制卡指令 ({label})**")
                        
                        # 提示语
                        st.info("💡 适用于：DeepSeek / ChatGPT / Claude / Gemini / Kimi 等任意 AI")
                        
                        # 生成指令
                        prompt = generate_ai_prompt(words)
                        
                        # === 关键优化：使用 st.code 实现一键复制 ===
                        # 以前是 text_area 需要全选，现在点击右上角图标即可
                        st.code(prompt, language='markdown')
                        st.caption("👆 点击右上角图标，一键复制完整指令，发送给 AI 即可。")

                with t1: show("target", "重点词")
                with t2: show("proper", "专有名词")
                with t3: show("beyond", "超纲词")
                with t4: show("known", "熟词")
            else:
                st.warning("无有效单词")