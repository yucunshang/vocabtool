import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import json
import time  # 新增：用于精确测速

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
    
    /* 优化数据看板外观 */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: var(--primary-color) !important;
    }
    
    /* 参数面板底色框 */
    .param-box {
        background-color: var(--secondary-background-color);
        padding: 15px 20px 5px 20px;
        border-radius: 10px;
        border: 1px solid var(--border-color-light);
        margin-bottom: 20px;
    }
    
    /* 复制提示文字高亮 */
    .copy-hint {
        color: #888;
        font-size: 14px;
        margin-bottom: 5px; 
        margin-top: 10px;
        padding-left: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据加载 (Data Loading)
# ==========================================
@st.cache_data
def load_knowledge_base():
    try:
        with open('data/terms.json', 'r', encoding='utf-8') as f:
            terms = json.load(f)
        with open('data/proper.json', 'r', encoding='utf-8') as f:
            proper = json.load(f)
        with open('data/patch.json', 'r', encoding='utf-8') as f:
            patch = json.load(f)
        with open('data/ambiguous.json', 'r', encoding='utf-8') as f:
            ambiguous = set(json.load(f))
            
        terms = {k.lower(): v for k, v in terms.items()}
        proper = {k.lower(): v for k, v in proper.items()}
        
        return terms, proper, patch, ambiguous
    except FileNotFoundError:
        st.error("⚠️ 缺少数据文件！请确保 `data/` 文件夹下包含 terms.json, proper.json, patch.json, ambiguous.json")
        return {}, {}, {}, set()

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

# ==========================================
# 3. 初始化 NLP (词形还原引擎)
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

def get_lemma(w):
    """提取单个单词的原型"""
    lemmas_dict = lemminflect.getAllLemmas(w)
    if not lemmas_dict:
        return w.lower()
    if 'ADJ' in lemmas_dict: return lemmas_dict['ADJ'][0]
    elif 'ADV' in lemmas_dict: return lemmas_dict['ADV'][0]
    elif 'VERB' in lemmas_dict: return lemmas_dict['VERB'][0]
    elif 'NOUN' in lemmas_dict: return lemmas_dict['NOUN'][0]
    else: return list(lemmas_dict.values())[0][0]

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
    
    for word, rank in BUILTIN_PATCH_VOCAB.items():
        vocab[word] = rank
        
    URGENT_OVERRIDES = {
        "china": 400, "turkey": 1500, "march": 500, "may": 100, "august": 1500, "polish": 2500,
        "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, "saturday": 300, "sunday": 300,
        "january": 400, "february": 400, "april": 400, "june": 400, "july": 400, "september": 400, "october": 400, "november": 400, "december": 400,
        "usa": 200, "uk": 200, "google": 1000, "apple": 1000, "microsoft": 1500
    }
    for word, rank in URGENT_OVERRIDES.items():
        vocab[word] = rank
        
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 5. AI 指令生成器
# ==========================================
def generate_ai_prompt(word_list, output_format, def_mode="single", is_term_list=False):
    words_str = ", ".join(word_list)
    core_principle_text = ""
    
    if is_term_list or def_mode == "term":
        core_principle_text = """1. 核心原则：领域锁定 (Domain Locked)
- **领域匹配**：如果单词带有 (Domain) 标签，**必须**仅提供符合该领域背景的专业释义。
- **原子性**：一张卡片只解释该领域的一个含义。"""
    elif def_mode == "split":
        core_principle_text = """1. 核心原则：原子性 (Atomicity)
- **含义拆分**：若一个单词有多个不同常用释义（名词 vs 动词，字面义 vs 引申义），**必须拆分为多条（1-3）独立数据**（即为同一个单词生成多行/多张卡片）。
- **严禁堆砌**：每张卡片只承载一个特定语境下的含义，不准将多个释义挤在一起。"""
    else: 
        core_principle_text = """1. 核心原则：极简速记 (Minimalist)
- **单一释义**：请**仅提供 1 个最核心、最常用的释义**。
- **严禁拆分**：对于这些生词，不要生成多张卡片，一张卡片即可。
- **减轻负担**：目的是快速混个脸熟，不要面面俱到。"""

    if output_format == 'csv':
        format_req = "CSV Code Block (后缀名 .csv)"
        format_desc = "请直接输出标准 CSV 代码块。"
    else:
        format_req = "TXT Code Block (后缀名 .txt)"
        format_desc = "请输出纯文本 TXT 代码块。"

    prompt = f"""
请扮演一位专业的 Anki 制卡专家。这是我整理的单词列表，请严格按照以下【核心原则】为我生成导入文件。

{core_principle_text}

2. 卡片正面 (Column 1: Front)
- 内容：提供自然的短语或搭配 (Phrase/Collocation)。
- 样式：纯文本。
- 注意：如果是“含义拆分”模式，正面可以是一样的单词/短语，但背面解释不同。

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
# 6. 核心分析引擎
# ==========================================
def analyze_words(unique_word_list):
    """直接对去重且还原后的单词列表进行词频定级"""
    unique_items = [] 
    JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're'}
    
    for item_lower in unique_word_list:
        if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
        if item_lower in JUNK_WORDS: continue
        
        actual_rank = vocab_dict.get(item_lower, 99999)
        
        # 1. 术语身份
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            unique_items.append({
                "word": f"{item_lower} ({domain})", 
                "rank": term_rank,
                "raw": item_lower
            })
            continue
        
        # 2. 专名与歧义词
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_lower.title())
            unique_items.append({
                "word": display,
                "rank": actual_rank, 
                "raw": item_lower
            })
            continue
            
        # 3. 普通词
        if actual_rank != 99999:
            unique_items.append({
                "word": item_lower,
                "rank": actual_rank,
                "raw": item_lower
            })
            
    return pd.DataFrame(unique_items)

# ==========================================
# 7. 界面布局与统一流水线
# ==========================================
st.title("🚀 Vocab Master Pro - 全能长文解析引擎")
st.markdown("💡 **一站式工作流**：支持粘贴几十万字的超长文本，**更支持直接上传 TXT 原著文件**，突破浏览器性能极限！系统将一键完成【词形还原】、【全量分级】并提取【Top N 精选】。")

if "raw_input_text" not in st.session_state:
    st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0 

def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 

# --- 参数配置区 ---
st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: current_level = st.number_input("🎯 当前水平 (起)", 0, 30000, 9000, 500, help="低于此词频的视为已掌握")
with c2: target_level = st.number_input("🎯 目标水平 (止)", 0, 30000, 15000, 500, help="高于此词频的视为超纲")
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 50, 10, help="从剩余生词中挑选的最核心数量")
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 3000, 500, help="精选时忽略太简单的基础词")
with c5: 
    st.write("") 
    st.write("") 
    show_rank = st.checkbox("🔢 附加显示 Rank", value=False)
st.markdown("</div>", unsafe_allow_html=True)

# --- 双通道输入区 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 在此粘贴英文原文 (支持10万字以内)...", height=200, key="raw_input_text")
with col_input2:
    st.info("💡 **突破极限**：超10万字的英文原著/论文，请勿粘贴，直接在此上传 👇")
    uploaded_file = st.file_uploader("📂 上传 .txt 纯文本文件", type=["txt"], key=f"uploader_{st.session_state.uploader_key}")

# --- 按钮区 ---
col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1:
    btn_process = st.button("🚀 一键智能解析 (处理长文)", type="primary", use_container_width=True)
with col_btn2:
    st.button("🗑️ 一键清空", on_click=clear_all_inputs, use_container_width=True)

st.divider()

combined_text = raw_text
if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    combined_text += "\n" + file_content

# --- 统一流水线处理逻辑 ---
if btn_process and combined_text.strip() and vocab_dict:
    # ⏱️ 记录开始时间
    start_time = time.time()
    
    with st.spinner("🧠 正在进行亿级词形还原与全量词频匹配（文件越大所需时间越长，请耐心等待）..."):
        
        # 1. 提取总词数
        raw_words = re.findall(r"[a-zA-Z']+", combined_text)
        total_word_count = len(raw_words)
        
        # 2. 智能还原
        lemmatized_words = [get_lemma(w) for w in raw_words]
        full_lemmatized_text = " ".join(lemmatized_words)
        
        # 3. 去重
        unique_lemmas = list(set([w.lower() for w in lemmatized_words]))
        unique_word_count = len(unique_lemmas)
        
        # 4. 分级
        df = analyze_words(unique_lemmas)
        valid_word_count = len(df)
        
        # ⏱️ 记录结束时间并计算耗时
        end_time = time.time()
        process_time = end_time - start_time
        
        # === 核心修改：新增第四列数据看板，展示闪电极速耗时 ===
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(label="📝 解析总字数", value=f"{total_word_count:,}")
        col_m2.metric(label="✂️ 去重词根数", value=f"{unique_word_count:,}")
        col_m3.metric(label="🎯 纳入分级词汇", value=f"{valid_word_count:,}")
        col_m4.metric(label="⚡ 极速解析耗时", value=f"{process_time:.2f} 秒")
        st.write("") # 留白
        
        if not df.empty:
            def categorize(row):
                r = row['rank']
                if r <= current_level: return "known"
                elif r <= target_level: return "target"
                else: return "beyond"
            
            df['final_cat'] = df.apply(categorize, axis=1)
            df = df.sort_values(by='rank')
            
            valid_candidates = df[df['rank'] >= min_rank_threshold].copy()
            top_df = valid_candidates.sort_values(by='rank', ascending=True).head(top_n)
            
            t_top, t_target, t_beyond, t_known, t_raw = st.tabs([
                f"🔥 Top {len(top_df)} 核心精选",
                f"🟡 重点 ({len(df[df['final_cat']=='target'])})", 
                f"🔴 超纲 ({len(df[df['final_cat']=='beyond'])})",
                f"🟢 已掌握 ({len(df[df['final_cat']=='known'])})",
                "📝 词形还原全文输出"
            ])
            
            def render_tab(tab_obj, data_df, label, def_mode, expand_default=False):
                with tab_obj:
                    if not data_df.empty:
                        pure_words = data_df['word'].tolist()
                        
                        display_lines = []
                        for _, row in data_df.iterrows():
                            if show_rank:
                                rank_str = str(int(row['rank'])) if row['rank'] != 99999 else "未收录"
                                display_lines.append(f"{row['word']} [Rank: {rank_str}]")
                            else:
                                display_lines.append(row['word'])
                        
                        with st.expander("👁️ 查看完整单词列表", expanded=expand_default):
                            st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制单词</p>", unsafe_allow_html=True)
                            st.code("\n".join(display_lines), language='text')
                        
                        st.markdown(f"**🤖 AI 指令 ({label})**")
                        has_term = any('(' in w for w in pure_words)
                        
                        p_csv = generate_ai_prompt(pure_words, 'csv', def_mode, is_term_list=has_term)
                        p_txt = generate_ai_prompt(pure_words, 'txt', def_mode, is_term_list=has_term)
                        
                        t_csv, t_txt = st.tabs(["📋 CSV 指令", "📝 TXT 指令"])
                        with t_csv: 
                            st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制指令</p>", unsafe_allow_html=True)
                            st.code(p_csv, language='markdown')
                        with t_txt: 
                            st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制指令</p>", unsafe_allow_html=True)
                            st.code(p_txt, language='markdown')
                    else: st.info("该区间暂无符合条件的单词")

            render_tab(t_top, top_df, "核心单义", def_mode="single", expand_default=True) 
            render_tab(t_target, df[df['final_cat']=='target'], "重点", def_mode="single", expand_default=False)
            render_tab(t_beyond, df[df['final_cat']=='beyond'], "超纲", def_mode="single", expand_default=False)
            render_tab(t_known, df[df['final_cat']=='known'], "熟词拆分", def_mode="split", expand_default=False)
            
            # 渲染还原原文板块 (防卡死 & 下载优化版)
            with t_raw:
                st.info("💡 这是自动词形还原（Lemmatized）后的全文。")
                
                # 1. 直接提供本地文件下载，完全绕过浏览器渲染限制！
                st.download_button(
                    label="💾 一键下载完整词形还原原文 (.txt)",
                    data=full_lemmatized_text,
                    file_name="lemmatized_full_text.txt",
                    mime="text/plain",
                    type="primary"
                )
                
                # 2. 限制网页端的显示长度（超过 5万字符 就截断防卡）
                display_limit = 50000
                if len(full_lemmatized_text) > display_limit:
                    st.warning("⚠️ 为防止浏览器卡死，网页仅展示前 50,000 个字符。请点击上方按钮下载完整版。")
                    st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内可复制展示部分</p>", unsafe_allow_html=True)
                    st.code(full_lemmatized_text[:display_limit] + "\n\n... [文本超长，剩余几十万字已折叠，请点击上方按钮下载查看] ...", language='text')
                else:
                    st.markdown("<p class='copy-hint'>👆 鼠标悬停在下方框内，点击右上角 📋 图标一键复制全文</p>", unsafe_allow_html=True)
                    st.code(full_lemmatized_text, language='text')