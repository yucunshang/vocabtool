import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    .preview-box { font-family: monospace; font-size: 12px; background: #f4f4f5; padding: 10px; border-radius: 5px; color: #666; max-height: 150px; overflow-y: auto; }
    
    /* 指南样式 */
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .guide-title { font-size: 18px; font-weight: bold; color: #0f172a; margin-bottom: 10px; display: block; }
    .guide-tip { font-size: 14px; color: #64748b; background: #eef2ff; padding: 8px; border-radius: 4px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源懒加载
# ==========================================
@st.cache_resource(show_spinner="正在加载 NLP 引擎...")
def load_nlp_resources():
    import nltk
    import lemminflect
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try: nltk.data.find(f'tokenizers/{pkg}')
            except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except: pass
    return nltk, lemminflect

def get_file_parsers():
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

def get_genanki():
    import genanki
    import tempfile
    return genanki, tempfile

@st.cache_data
def load_vocab_data():
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
        except: return {}, None
    return {}, None

VOCAB_DICT, FULL_DF = load_vocab_data()

def get_beijing_time_str():
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

def clear_all_state():
    st.session_state.clear()

# ==========================================
# 2. 核心逻辑
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'epub':
            genanki, tempfile = get_genanki()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + " "
            os.remove(tmp_path)
    except Exception as e:
        return f"Error: {e}"
    return text

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    nltk, lemminflect = load_nlp_resources()
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    # 1. 提取所有单词
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    total_words = len(raw_tokens)
    
    # 2. 原始去重
    unique_tokens = set(raw_tokens)
    
    target_words = []
    seen_lemmas = set() # 3. 词根去重 (go/went 问题)
    
    for w in unique_tokens:
        if len(w) < 2: continue 
        lemma = get_lemma_local(w)
        
        if lemma in seen_lemmas: continue
            
        rank = VOCAB_DICT.get(lemma, 99999)
        
        # 4. 筛选逻辑：区间内 OR (是生词 且 允许生词)
        is_in_range = (rank >= current_lvl and rank <= target_lvl)
        is_unknown_included = (rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            target_words.append((lemma, rank))
            seen_lemmas.add(lemma)
            
    # 未知词排在最后
    target_words.sort(key=lambda x: x[1])
    return [x[0] for x in target_words], total_words

def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.replace("```json", "").replace("```", "").strip()
    # 正则流式解析，不惧换行
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases = set()

    for match in matches:
        json_str = match.group()
        try:
            data = json.loads(json_str, strict=False)
            front_text = data.get("w", "").strip()
            meaning = data.get("m", "").strip()
            examples = data.get("e", "").strip()
            etymology = data.get("r", "").strip()
            
            if not etymology or etymology.lower() == "none" or etymology == "":
                etymology = ""

            if not front_text or not meaning: continue
            front_text = front_text.replace('**', '')
            if front_text in seen_phrases: continue
            seen_phrases.add(front_text)

            parsed_cards.append({
                'front_phrase': front_text,
                'meaning': meaning,
                'examples': examples,
                'etymology': etymology
            })
        except: continue
    return parsed_cards

# ==========================================
# 3. Anki 生成
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
    
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; line-height: 1.3; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .nightMode .definition { color: #e0e0e0; }
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 18px; line-height: 1.5; margin-bottom: 15px; text-align: left; }
    .nightMode .examples { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .footer-info { margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 10px; text-align: left; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; line-height: 1.4; border: 1px solid #fef3c7; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id, f'VocabFlow JSON Model {model_id}',
        fields=[{'name': 'FrontPhrase'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{FrontPhrase}}</div>', 
            'afmt': '''
            {{FrontSide}}<hr>
            <div class="definition">{{Meaning}}</div>
            <div class="examples">{{Examples}}</div>
            {{#Etymology}}
            <div class="footer-info"><div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div></div>
            {{/Etymology}}
            ''',
        }], css=CSS
    )
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[str(c['front_phrase']), str(c['meaning']), str(c['examples']).replace('\n','<br>'), str(c['etymology'])]))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt Logic
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    if front_mode == "单词 (Word)":
        w_instr = "Key `w`: The word itself (lowercase)."
    else:
        w_instr = "Key `w`: A short practical collocation/phrase (2-5 words)."

    if def_mode == "中文":
        m_instr = "Key `m`: Concise Chinese definition (max 10 chars)."
    elif def_mode == "中英双语":
        m_instr = "Key `m`: English Definition + Chinese Definition."
    else:
        m_instr = "Key `m`: English definition (concise)."

    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."

    if need_ety:
        r_instr = "Key `r`: Simplified Chinese Etymology (Root/Prefix)."
    else:
        r_instr = "Key `r`: Leave this empty string \"\"."

    return f"""
Task: Create Anki cards.
Words: {w_list}

**OUTPUT: NDJSON (One line per object).**

**Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Keys:** `w` (Front), `m` (Meaning), `e` (Examples), `r` (Etymology)

**Example:**
{{"w": "...", "m": "...", "e": "...", "r": "..."}}

**Start:**
"""

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra (V25)")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南 (完整版)", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

with tab_guide:
    st.markdown("""
    ### 👋 欢迎使用 Vocab Flow Ultra
    这是一个**从阅读材料中提取生词**，并利用 **AI** 自动生成 **Anki 卡片**的效率工具。
    
    ---
    
    <div class="guide-step">
    <span class="guide-title">Step 1: 提取生词 (Extract)</span>
    在 <code>1️⃣ 单词提取</code> 标签页：<br><br>
    <strong>1. 上传文件</strong><br>
    支持 <code>.pdf</code>, <code>.txt</code>, <code>.epub</code>, <code>.docx</code>，或者直接粘贴文本。<br>
    <div class="guide-tip">💡 系统会自动过滤掉文档中的非单词字符，并将 <code>went</code>, <code>goes</code> 还原为 <code>go</code> 进行统计。</div>
    <br>
    <strong>2. 设置过滤范围 (Rank Filter)</strong><br>
    利用 COCA 20000 词频表进行科学筛选：
    <ul>
        <li><strong>忽略排名前 N</strong> (Min Rank)：例如设为 <code>2000</code>，会过滤掉 `the, is, you` 等最基础的高频词。如果你基础很好，可以设为 <code>5000</code>。</li>
        <li><strong>忽略排名后 N</strong> (Max Rank)：例如设为 <code>15000</code>，会过滤掉极其生僻的词。</li>
        <li><strong>🔓 包含生僻词</strong> (Unknown)：《冰与火之歌》等小说包含大量人名或自造词，它们没有排名。勾选此项可以强制提取它们。</li>
    </ul>
    <br>
    <strong>3. 点击 🚀 开始分析</strong><br>
    系统会显示“数据看板”，告诉你文档总字数、筛选出了多少生词，以及耗时。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 2: 获取 Prompt (AI Generation)</span>
    分析完成后，你会看到生成的单词列表。<br><br>
    <strong>1. 自定义设置 (Customize)</strong><br>
    点击 <code>⚙️ 自定义 Prompt 设置</code> 展开面板：
    <ul>
        <li><strong>正面内容</strong>：选择背单词本身，还是背短语搭配。</li>
        <li><strong>背面释义</strong>：英文、中文或中英双语。</li>
        <li><strong>例句/词源</strong>：按需开启。</li>
    </ul>
    <br>
    <strong>2. 复制 Prompt</strong><br>
    系统会自动将单词分组（防止 AI 长度溢出）。
    <ul>
        <li>📱 <strong>手机端</strong>：使用下方的“纯文本框”，长按全选 -> 复制。</li>
        <li>💻 <strong>电脑端</strong>：点击代码块右上角的 Copy 📄 图标。</li>
    </ul>
    <br>
    <strong>3. 发送给 AI</strong><br>
    将复制的内容发送给 ChatGPT / Claude / Gemini / DeepSeek。AI 会返回一串 JSON 数据。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 3: 制作 Anki 牌组 (Create Deck)</span>
    在 <code>2️⃣ Anki 制作</code> 标签页：<br><br>
    <strong>1. 粘贴 AI 回复</strong><br>
    将 AI 生成的 JSON 内容粘贴到输入框中。<br>
    <div class="guide-tip">💡 <strong>支持追加粘贴</strong>：如果你有 5 组单词，可以把 AI 的 5 次回复依次粘贴在同一个框里，不需要分批下载。</div>
    <br>
    <strong>2. 下载与导入</strong><br>
    点击 <strong>📥 下载 .apkg</strong>，然后双击该文件，它会自动导入到你的 Anki 软件中。
    </div>
    
    <div class="guide-step" style="border-left-color: #10b981; background-color: #ecfdf5;">
    <span class="guide-title">💡 进阶技巧</span>
    <ul>
        <li><strong>一键复制所有单词</strong>：在“分析报告”下方，有一个“全部生词”的代码块，点击右上角图标可一次性导出到 Excel。</li>
        <li><strong>文件读取检查</strong>：如果觉得提取的词太少，可以展开“🔍 文件读取验尸”查看文档开头和结尾，确认程序是否读完了整本书。</li>
        <li><strong>词根去重</strong>：V23 版本已升级去重算法，不会再同时出现 <code>go</code> 和 <code>went</code>。</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析", "🔢 词频列表"])
    
    with mode_context:
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略排名前 N 的词", 1, 20000, 100, step=100)
        targ = c2.number_input("忽略排名后 N 的词", 2000, 50000, 20000, step=500)
        
        include_unknown = st.checkbox("🔓 包含词典里没有的生僻词/人名 (Rank > 20000)", value=False, help="《冰与火之歌》等奇幻小说有很多自造词，勾选此项可以提取它们。")
        
        uploaded_file = st.file_uploader("📂 上传文档 (TXT/PDF/DOCX/EPUB)")
        pasted_text = st.text_area("📄 ...或粘贴文本", height=100)
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 读取文件...")
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 10:
                    status.write(f"🔍 提取 {len(raw_text)} 字符，加载 NLP 库...")
                    final_words, raw_count = analyze_logic(raw_text, curr, targ, include_unknown)
                    
                    st.session_state['gen_words'] = final_words
                    st.session_state['raw_count'] = raw_count
                    st.session_state['process_time'] = time.time() - start_time
                    st.session_state['raw_text_preview'] = raw_text 
                    
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容太短", state="error")
        
        if st.button("🗑️ 清空", type="secondary", on_click=clear_all_state): pass

    with mode_rank:
        gen_type = st.radio("模式", ["🔢 顺序", "🔀 随机"], horizontal=True)
        if "顺序" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
             count = c_b.number_input("数量", 10, 500, 50, step=10)
             if st.button("🚀 生成"):
                 start_time = time.time()
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                     st.session_state['gen_words'] = subset[w_col].tolist()
                     st.session_state['raw_count'] = 0
                     st.session_state['process_time'] = time.time() - start_time
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("Min Rank", 1, 20000, 1, step=100)
             max_r = c_max.number_input("Max Rank", 1, 25000, 5000, step=100)
             r_count = c_cnt.number_input("Count", 10, 200, 50, step=10)
             if st.button("🎲 抽取"):
                 start_time = time.time()
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                     candidates = FULL_DF[mask]
                     if len(candidates) > 0:
                         subset = candidates.sample(n=min(r_count, len(candidates))).sort_values(r_col)
                         st.session_state['gen_words'] = subset[w_col].tolist()
                         st.session_state['raw_count'] = 0
                         st.session_state['process_time'] = time.time() - start_time

    if 'gen_words' in st.session_state and st.session_state['gen_words']:
        words = st.session_state['gen_words']
        
        st.divider()
        st.markdown("### 📊 分析报告")
        
        with st.expander("🔍 **文件读取验尸 (Check First/Last 500 chars)**"):
            raw_preview = st.session_state.get('raw_text_preview', "")
            if raw_preview:
                st.markdown("**Head (开头 500 字符):**")
                st.markdown(f"<div class='preview-box'>{raw_preview[:500]}...</div>", unsafe_allow_html=True)
                st.markdown("**Tail (结尾 500 字符):**")
                st.markdown(f"<div class='preview-box'>...{raw_preview[-500:]}</div>", unsafe_allow_html=True)
            else:
                st.info("无原文档数据（词频模式或未上传）。")

        k1, k2, k3 = st.columns(3)
        raw_c = st.session_state.get('raw_count', 0)
        p_time = st.session_state.get('process_time', 0.1)
        k1.metric("📄 文档总字数", f"{raw_c:,}")
        k2.metric("🎯 筛选生词", f"{len(words)}")
        k3.metric("⚡ 耗时", f"{p_time:.2f}s")
        
        st.markdown("### 📋 全部生词 (点击右上角复制)")
        all_words_str = ", ".join(words)
        st.code(all_words_str, language="text")

        with st.expander("⚙️ **自定义 Prompt 设置 (点击展开)**", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("正面内容", ["短语搭配 (Phrase)", "单词 (Word)"])
            def_mode = col_s2.selectbox("背面释义", ["英文", "中文", "中英双语"])
            
            col_s3, col_s4 = st.columns(2)
            ex_count = col_s3.slider("例句数量", 1, 3, 1)
            need_ety = col_s4.checkbox("包含词源/词根", value=True)

        batch_size = st.number_input("AI 分组大小", 10, 200, 100, step=10)
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 第 {idx+1} 组 (共 {len(batch)} 词)", expanded=(idx==0)):
                prompt_text = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.caption("📱 手机端专用：")
                st.text_area(f"text_area_{idx}", value=prompt_text, height=100, label_visibility="collapsed")
                st.caption("💻 电脑端：")
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 📦 制作 Anki")
    bj_time_str = get_beijing_time_str()
    if 'anki_input_text' not in st.session_state: st.session_state['anki_input_text'] = ""

    st.caption("👇 粘贴 AI 回复：")
    ai_resp = st.text_area("JSON 输入框", height=300, key="anki_input_text")
    deck_name = st.text_input("牌组名", f"Vocab_{bj_time_str}")
    
    if ai_resp.strip():
        parsed_data = parse_anki_data(ai_resp)
        if parsed_data:
            st.success(f"✅ 成功解析 {len(parsed_data)} 条数据")
            df_view = pd.DataFrame(parsed_data)
            df_view.rename(columns={'front_phrase': '正面', 'meaning': '背面', 'etymology': '词源'}, inplace=True)
            st.dataframe(df_view[['正面', '背面', '词源']], use_container_width=True, hide_index=True)
            
            f_path = generate_anki_package(parsed_data, deck_name)
            with open(f_path, "rb") as f:
                st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
        else:
            st.warning("⚠️ 等待粘贴...")