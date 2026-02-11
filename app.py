import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import json
import time
import requests

# 尝试导入多格式文档处理库，如果没有则提示
try:
    import PyPDF2
    import docx
except ImportError:
    st.error("⚠️ 缺少文件处理依赖。请在终端运行: pip install PyPDF2 python-docx")

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 16px !important; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    [data-testid="stMetricValue"] { font-size: 28px !important; color: var(--primary-color) !important; }
    .param-box { background-color: var(--secondary-background-color); padding: 15px 20px 5px 20px; border-radius: 10px; border: 1px solid var(--border-color-light); margin-bottom: 20px; }
    .copy-hint { color: #888; font-size: 14px; margin-bottom: 5px; margin-top: 10px; padding-left: 5px; }
    .exam-tag { font-size: 12px; background: #e0e0e0; color: #333; padding: 2px 6px; border-radius: 4px; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据与 NLP 初始化
# ==========================================
@st.cache_data
def load_knowledge_base():
    try:
        with open('data/terms.json', 'r', encoding='utf-8') as f: terms = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/proper.json', 'r', encoding='utf-8') as f: proper = {k.lower(): v for k, v in json.load(f).items()}
        with open('data/patch.json', 'r', encoding='utf-8') as f: patch = json.load(f)
        with open('data/ambiguous.json', 'r', encoding='utf-8') as f: ambiguous = set(json.load(f))
        return terms, proper, patch, ambiguous
    except FileNotFoundError:
        st.error("⚠️ 缺少 data/ 文件夹下的 JSON 知识库文件！")
        return {}, {}, {}, set()

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_knowledge_base()

@st.cache_resource
def setup_nltk():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    for pkg in ['averaged_perceptron_tagger', 'punkt']:
        try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except: pass
setup_nltk()

def get_lemma(w):
    """提取词根 (更细粒度)"""
    lemmas_dict = lemminflect.getAllLemmas(w)
    if not lemmas_dict: return w.lower()
    for pos in ['ADJ', 'ADV', 'VERB', 'NOUN']:
        if pos in lemmas_dict: return lemmas_dict[pos][0]
    return list(lemmas_dict.values())[0][0]

@st.cache_data
def load_vocab():
    vocab = {}
    file_path = next((f for f in ["coca_cleaned.csv", "data.csv"] if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
            df = df.sort_values(r_col, ascending=True).drop_duplicates(subset=[w_col], keep='first')
            vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except: pass
    
    for word, rank in BUILTIN_PATCH_VOCAB.items(): vocab[word] = rank
    URGENT_OVERRIDES = {
        "china": 400, "turkey": 1500, "march": 500, "may": 100, "august": 1500, "polish": 2500,
        "monday": 300, "tuesday": 300, "wednesday": 300, "thursday": 300, "friday": 300, "saturday": 300, "sunday": 300,
        "january": 400, "february": 400, "april": 400, "june": 400, "july": 400, "september": 400, "october": 400, "november": 400, "december": 400,
        "usa": 200, "uk": 200, "google": 1000, "apple": 1000, "microsoft": 1500
    }
    for word, rank in URGENT_OVERRIDES.items(): vocab[word] = rank
    return vocab

vocab_dict = load_vocab()

# ==========================================
# 3. 核心功能映射：考试大纲 & AI
# ==========================================
def get_exam_syllabus(rank):
    """内置 COCA Rank 到 国内外考试大纲 的映射关系"""
    if rank == 99999: return "未收录/超纲"
    if rank <= 1500: return "小学/初中"
    if rank <= 3500: return "中考核心"
    if rank <= 5500: return "高考核心"
    if rank <= 7500: return "CET-4 (四级)"
    if rank <= 9500: return "CET-6 (六级)"
    if rank <= 13000: return "考研/雅思"
    if rank <= 20000: return "托福/GRE"
    return "极难词汇"

def extract_text_from_file(uploaded_file):
    """支持 txt, pdf, docx 多种格式解析"""
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == 'txt':
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ext == 'pdf':
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            doc = docx.Document(uploaded_file)
            return " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.error(f"文件解析失败: {e}")
        return ""
    return ""

def call_deepseek_api(api_key, prompt_template, words):
    """调用 DeepSeek 接口直接生成制卡 CSV"""
    if not api_key: return "⚠️ 错误：未提供 API Key 或 管理员密码。"
    if not words: return "⚠️ 错误：没有需要生成的单词。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    full_prompt = f"{prompt_template}\n\n待处理单词：\n{', '.join(words)}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"🚨 API 调用失败，请检查网络或 Key 是否正确。\n详细错误: {str(e)}"

# ==========================================
# 4. 分析引擎
# ==========================================
def analyze_words(unique_word_list):
    unique_items = [] 
    JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're'}
    for item_lower in unique_word_list:
        if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
        if item_lower in JUNK_WORDS: continue
        actual_rank = vocab_dict.get(item_lower, 99999)
        
        syllabus = get_exam_syllabus(actual_rank if actual_rank != 99999 else 99999)
        
        if item_lower in BUILTIN_TECHNICAL_TERMS:
            domain = BUILTIN_TECHNICAL_TERMS[item_lower]
            term_rank = actual_rank if actual_rank != 99999 else 15000
            unique_items.append({"word": f"{item_lower} ({domain})", "rank": term_rank, "raw": item_lower, "syllabus": "专业术语"})
            continue
        
        if item_lower in PROPER_NOUNS_DB or item_lower in AMBIGUOUS_WORDS:
            display = PROPER_NOUNS_DB.get(item_lower, item_lower.title())
            unique_items.append({"word": display, "rank": actual_rank, "raw": item_lower, "syllabus": "专有名词"})
            continue
            
        if actual_rank != 99999:
            unique_items.append({"word": item_lower, "rank": actual_rank, "raw": item_lower, "syllabus": syllabus})
            
    return pd.DataFrame(unique_items)

# ==========================================
# 5. UI 与流水线
# ==========================================
st.title("🚀 Vocab Master Pro - 全能智能教研引擎")
st.markdown("💡 支持粘贴长文或上传 `TXT / PDF / DOCX`，自动大纲映射，并内置 **DeepSeek AI** 一键生成 Anki 记忆卡片。")

if "raw_input_text" not in st.session_state: st.session_state.raw_input_text = ""
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
def clear_all_inputs():
    st.session_state.raw_input_text = ""
    st.session_state.uploader_key += 1 

# --- 参数配置区 ---
st.markdown("<div class='param-box'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: current_level = st.number_input("🎯 当前水平 (起)", 0, 30000, 7500, 500, help="低于此词频的视为已掌握")
with c2: target_level = st.number_input("🎯 目标水平 (止)", 0, 30000, 15000, 500, help="高于此词频的视为超纲")
with c3: top_n = st.number_input("🔥 精选 Top N", 10, 500, 50, 10)
with c4: min_rank_threshold = st.number_input("📉 忽略前 N 词", 0, 20000, 3500, 500)
with c5: 
    st.write("") 
    st.write("") 
    show_visual = st.checkbox("📊 显示可视化反馈", value=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- 双通道多格式输入 ---
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本 (支持10万字以内)", height=150, key="raw_input_text")
with col_input2:
    st.info("💡 **多格式解析**：支持超大 `.txt`, `.pdf`, `.docx` 原著文件 👇")
    uploaded_file = st.file_uploader("📂 上传文档", type=["txt", "pdf", "docx"], key=f"uploader_{st.session_state.uploader_key}")

col_btn1, col_btn2 = st.columns([5, 1])
with col_btn1: btn_process = st.button("🚀 极速智能解析", type="primary", use_container_width=True)
with col_btn2: st.button("🗑️ 一键清空", on_click=clear_all_inputs, use_container_width=True)

st.divider()

combined_text = raw_text
if uploaded_file is not None:
    combined_text += "\n" + extract_text_from_file(uploaded_file)

if btn_process and combined_text.strip() and vocab_dict:
    start_time = time.time()
    
    with st.spinner("🧠 正在进行多线程词汇拆解与大纲映射..."):
        raw_words = re.findall(r"[a-zA-Z']+", combined_text)
        lemmatized_words = [get_lemma(w) for w in raw_words]
        full_lemmatized_text = " ".join(lemmatized_words)
        
        unique_lemmas = list(set([w.lower() for w in lemmatized_words]))
        df = analyze_words(unique_lemmas)
        
        process_time = time.time() - start_time
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(label="📝 解析总字数", value=f"{len(raw_words):,}")
        col_m2.metric(label="✂️ 去重词根数", value=f"{len(unique_lemmas):,}")
        col_m3.metric(label="🎯 纳入分级词汇", value=f"{len(df):,}")
        col_m4.metric(label="⚡ 极速解析耗时", value=f"{process_time:.2f} 秒")
        
        if not df.empty:
            # === 可视化反馈区 (可选) ===
            if show_visual:
                st.subheader("📊 词汇分布大纲雷达图")
                chart_data = df['syllabus'].value_counts()
                st.bar_chart(chart_data, color="#ff4b4b")
                st.caption("👆 通过上图可直观判断这篇文章对应国内哪种考试难度。")
                st.divider()
            
            def categorize(row):
                r = row['rank']
                if r <= current_level: return "known"
                elif r <= target_level: return "target"
                else: return "beyond"
            
            df['final_cat'] = df.apply(categorize, axis=1)
            df = df.sort_values(by='rank')
            top_df = df[df['rank'] >= min_rank_threshold].sort_values(by='rank', ascending=True).head(top_n)
            
            t_top, t_target, t_beyond, t_known, t_raw = st.tabs([
                f"🔥 Top {len(top_df)}", f"🟡 重点 ({len(df[df['final_cat']=='target'])})", 
                f"🔴 超纲 ({len(df[df['final_cat']=='beyond'])})", f"🟢 已掌握 ({len(df[df['final_cat']=='known'])})",
                "📝 原文防卡死下载"
            ])
            
            # --- AI 动态 Prompt 定义 ---
            default_prompt = """请扮演一位专业的 Anki 制卡专家。请严格为以下单词生成 CSV 导入格式。
核心原则：
1. 极简速记：仅提供1个最核心、最符合现代语境的释义。
2. 结构(每字段用英文逗号分隔，内容加双引号)："单词或短语", "英文释义<br><br><em>斜体例句</em><br><br>中文助记"
请直接输出标准 CSV 代码块，不要包含任何多余解释。"""

            def render_tab(tab_obj, data_df, label, expand_default=False, df_key=""):
                with tab_obj:
                    if not data_df.empty:
                        pure_words = data_df['word'].tolist()
                        
                        # 展示大纲映射标签
                        display_lines = []
                        for _, row in data_df.iterrows():
                            rank_str = str(int(row['rank'])) if row['rank'] != 99999 else "未收录"
                            display_lines.append(f"{row['word']} [Rank: {rank_str}] - 【{row['syllabus']}】")
                        
                        with st.expander("👁️ 查看带有大纲映射的单词列表", expanded=expand_default):
                            st.code("\n".join(display_lines), language='text')
                        
                        # ==========================================
                        # 🤖 原生内置 DeepSeek AI 引擎 (安全鉴权版)
                        # ==========================================
                        st.markdown(f"#### 🤖 AI 一键制卡引擎 ({label})")
                        
                        col_ai1, col_ai2 = st.columns([1, 1])
                        with col_ai1:
                            ai_pwd = st.text_input("🔑 鉴权密码 / API Key", type="password", placeholder="输入站长密码或您自己的 DeepSeek Key", key=f"pwd_{df_key}")
                        with col_ai2:
                            st.write("")
                            st.write("")
                            st.caption("访客必须自备 Key；站长输入特权密码即可直接调用内置额度。")
                        
                        custom_prompt = st.text_area("📝 自定义 AI Prompt (可动态修改)", value=default_prompt, height=150, key=f"prompt_{df_key}")
                        
                        if st.button("⚡ 召唤 DeepSeek 立即生成 CSV", key=f"btn_{df_key}", type="primary"):
                            with st.spinner("AI 正在光速编纂卡片，请稍候..."):
                                # --- 核心鉴权逻辑 ---
                                actual_key = ""
                                try:
                                    # 如果输入的密码等于后台设置的站长密码，则提取隐藏的 API Key
                                    if ai_pwd == st.secrets["APP_PASSWORD"]:
                                        actual_key = st.secrets["DEEPSEEK_API_KEY"]
                                    else:
                                        # 否则，把用户输入的当成他们自己的 API Key
                                        actual_key = ai_pwd
                                except:
                                    # 本地测试如果没有 secrets 文件，直接使用输入的字符串
                                    actual_key = ai_pwd
                                
                                ai_result = call_deepseek_api(actual_key, custom_prompt, pure_words)
                                
                                st.success("🎉 生成完成！")
                                st.code(ai_result, language="markdown")
                                
                                # 支持直接把 AI 结果存成 CSV 文件下载
                                st.download_button(
                                    label="📥 直接下载生成的 Anki 卡片 (.csv)",
                                    data=ai_result,
                                    file_name=f"anki_cards_{label}.csv",
                                    mime="text/csv"
                                )
                    else: st.info("该区间暂无单词")

            render_tab(t_top, top_df, "Top精选", expand_default=True, df_key="top") 
            render_tab(t_target, df[df['final_cat']=='target'], "重点", expand_default=False, df_key="target")
            render_tab(t_beyond, df[df['final_cat']=='beyond'], "超纲", expand_default=False, df_key="beyond")
            render_tab(t_known, df[df['final_cat']=='known'], "熟词", expand_default=False, df_key="known")
            
            with t_raw:
                st.info("💡 这是自动词形还原后的全文输出，已针对长文优化防卡死体验。")
                st.download_button(label="💾 一键下载完整词形还原原文 (.txt)", data=full_lemmatized_text, file_name="lemmatized_text.txt", mime="text/plain", type="primary")
                if len(full_lemmatized_text) > 50000:
                    st.warning("⚠️ 文本超长，仅展示前 50,000 字符。")
                    st.code(full_lemmatized_text[:50000] + "\n\n... [请下载查看完整内容] ...", language='text')
                else:
                    st.code(full_lemmatized_text, language='text')