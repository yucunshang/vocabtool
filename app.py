# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置 (Page Config)
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra",
    page_icon="⚡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    .scrollable-text {
        max-height: 250px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        background-color: #fafafa;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 13px;
    }
    .guide-step { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #0056b3; }
    .guide-title { font-weight: bold; color: #0f172a; display: block; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

# ==========================================
# 1. 资源懒加载 (Resource Loading)
# ==========================================

@st.cache_resource(show_spinner="正在加载 NLP 引擎 (首次运行较慢)...")
def load_nlp_resources():
    """
    懒加载 NLTK 和 Lemminflect 以提升启动速度
    """
    import nltk
    import lemminflect
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # 需要的 NLTK 包
    required_packages = ['averaged_perceptron_tagger', 'punkt', 'punkt_tab', 'wordnet']
    
    for pkg in required_packages:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{pkg}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{pkg}')
                except LookupError:
                    nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    
    return nltk, lemminflect

@st.cache_data
def load_vocab_data():
    """
    加载 COCA 词频表。返回 {word: rank} 字典和完整 DataFrame。
    """
    # 文件名优先级
    possible_files = ["coca_cleaned.csv", "vocab.csv", "data.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # 规范化列名
            df.columns = [c.strip().lower() for c in df.columns]
            
            # 动态识别列
            w_col = next((c for c in df.columns if 'word' in c), None)
            r_col = next((c for c in df.columns if 'rank' in c), None)
            
            if not w_col or not r_col:
                return {}, None

            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            
            # 去重：保留排名最高的（数值最小的）
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            return vocab_dict, df
        except Exception as e:
            st.error(f"读取词汇表错误: {e}")
            return {}, None
    return {}, None

# 全局加载一次数据
VOCAB_DICT, FULL_DF = load_vocab_data()

def get_beijing_time_str():
    """获取格式化的北京时间字符串"""
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

def clear_all_state():
    """强制重置 Session State"""
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'anki_input_text']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    # 随机化 uploader key 以强制重置上传组件
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. 核心逻辑：提取与分析 (Core Logic)
# ==========================================

def extract_text_from_file(uploaded_file):
    """解析 PDF, DOCX, EPUB, TXT"""
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    
    text = ""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'txt':
            bytes_data = uploaded_file.getvalue()
            # 尝试常见编码
            for encoding in ['utf-8', 'gb18030', 'latin-1']:
                try:
                    text = bytes_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
        elif file_ext == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text_parts = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
            text = "\n".join(text_parts)
            
        elif file_ext == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
            
        elif file_ext == 'epub':
            # 处理 EPUB 需要临时文件
            with open("temp.epub", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            book = epub.read_epub("temp.epub")
            text_parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text_parts.append(soup.get_text(separator=' ', strip=True))
            text = " ".join(text_parts)
            if os.path.exists("temp.epub"):
                os.remove("temp.epub")
                
    except Exception as e:
        return f"读取文件错误: {str(e)}"
        
    return text

def is_valid_word(word):
    """启发式清洗：去除垃圾词"""
    if len(word) < 2: return False
    if len(word) > 25: return False
    # 过滤连续3个相同字符
    if re.search(r'(.)\1{2,}', word): return False
    # 必须包含至少一个元音 (英文启发式规则)
    if not re.search(r'[aeiouy]', word): return False
    # 不允许包含数字或下划线
    if re.search(r'[0-9_]', word): return False
    return True

def analyze_logic(text, min_rank, max_rank, include_unknown):
    """
    核心算法: 分词 -> 词形还原 -> 排名检查 -> 去重
    返回: [(word, rank), ...], raw_word_count
    """
    nltk, lemminflect = load_nlp_resources()
    
    # 1. 分词 (保留内部连字符如 'well-known')
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(raw_tokens)
    
    # 2. 初步清洗
    clean_tokens = set([t.lower() for t in raw_tokens if is_valid_word(t.lower())])
    
    final_candidates = [] 
    seen_lemmas = set()
    
    for w in clean_tokens:
        # 获取 Lemma (词原)，例如 went -> go
        # 优先尝试 VERB，因为变化最多
        try:
            lemma = lemminflect.getLemma(w, upos='VERB')[0]
        except:
            lemma = w
            
        # 获取排名
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        # 确定有效排名 (取两者中较小/靠前的)
        best_rank = min(rank_lemma, rank_orig)
        
        # 确定输出单词 (如果 Lemma 有效则优先输出 Lemma)
        word_to_keep = lemma if rank_lemma != 99999 else w
        
        # 过滤逻辑
        is_in_range = (min_rank <= best_rank <= max_rank)
        is_unknown_included = (include_unknown and best_rank == 99999)
        
        if is_in_range or is_unknown_included:
            # 去重：使用 lemma 作为 key
            # 确保 'go' 和 'went' 不会同时出现
            dedupe_key = lemma
            
            if dedupe_key not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(dedupe_key)
    
    # 排序: 高频(低 rank) -> 低频 -> 未知
    final_candidates.sort(key=lambda x: x[1])
    
    return final_candidates, total_words

# ==========================================
# 3. Anki 解析与生成 (Anki Generation)
# ==========================================

def parse_anki_data(raw_text):
    """
    从 AI 回复中提取 JSON 对象。
    输入: 可能包含 markdown、文本和多个 JSON 对象的字符串。
    输出: 字典列表。
    """
    parsed_cards = []
    # 移除 markdown 代码块标记
    text = raw_text.replace("```json", "").replace("```", "").strip()
    
    # 正则匹配 JSON 结构 { ... }
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases = set()

    for match in matches:
        json_str = match.group()
        try:
            data = json.loads(json_str, strict=False)
            
            # 提取字段，带默认值
            front = str(data.get("w", "")).strip()
            meaning = str(data.get("m", "")).strip()
            examples = str(data.get("e", "")).strip()
            etymology = str(data.get("r", "")).strip()
            
            if etymology.lower() in ["none", "", "null"]:
                etymology = ""

            # 基础验证
            if not front or not meaning:
                continue
            
            # 移除正面可能存在的 markdown 加粗
            front = front.replace('**', '')
            
            # 批次内去重
            if front.lower() in seen_phrases:
                continue
            seen_phrases.add(front.lower())

            parsed_cards.append({
                'front': front,
                'back': meaning,
                'examples': examples,
                'etymology': etymology
            })
        except json.JSONDecodeError:
            continue
            
    return parsed_cards

def generate_anki_package(cards_data, deck_name):
    """使用 genanki 生成 .apkg 文件"""
    import genanki
    import tempfile
    
    # 卡片 CSS 样式
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .nightMode .definition { color: #e0e0e0; }
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; font-style: italic; font-size: 18px; text-align: left; margin-bottom: 15px; }
    .nightMode .examples { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .etymology { font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border: 1px solid #fef3c7; border-radius: 6px; text-align: left; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    
    # 创建唯一 Model ID
    model_id = random.randrange(1 << 30, 1 << 31)
    
    model = genanki.Model(
        model_id,
        f'VocabFlow Model {model_id}',
        fields=[
            {'name': 'Front'}, 
            {'name': 'Meaning'}, 
            {'name': 'Examples'}, 
            {'name': 'Etymology'}
        ],
        templates=[{
            'name': 'Standard Card',
            'qfmt': '<div class="phrase">{{Front}}</div>', 
            'afmt': '''
            {{FrontSide}}<hr>
            <div class="definition">{{Meaning}}</div>
            <div class="examples">{{Examples}}</div>
            {{#Etymology}}
            <div class="etymology">🌱 <b>Origin:</b> {{Etymology}}</div>
            {{/Etymology}}
            ''',
        }],
        css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    
    for c in cards_data:
        note = genanki.Note(
            model=model,
            fields=[
                c['front'], 
                c['back'], 
                c['examples'].replace('\n','<br>'), 
                c['etymology']
            ]
        )
        deck.add_note(note)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt Engineering (提示词生成)
# ==========================================

def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    # 可配置的指令
    w_instr = "Key `w`: The word itself (lemma)." if "单词" in front_mode else "Key `w`: A common short phrase/collocation using the word."
    
    if def_mode == "中文":
        m_instr = "Key `m`: Concise Chinese definition."
    elif def_mode == "中英双语":
        m_instr = "Key `m`: English Definition <br> Chinese Definition."
    else:
        m_instr = "Key `m`: Simple English definition."

    e_instr = f"Key `e`: {ex_count} native example sentence(s). Use <br> for line breaks."
    r_instr = "Key `r`: Etymology/Root explanation (in Chinese)." if need_ety else "Key `r`: Empty string."

    return f"""
Task: Create high-quality Anki flashcards (JSON format).
Words to process: {w_list}

**Format:** NDJSON (Newline Delimited JSON). Do not use lists. One JSON object per line.

**Field Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Output keys:** `w`, `m`, `e`, `r`

**Example:**
{{"w": "example", "m": "an instance serving to illustrate", "e": "This is a good example.", "r": "from Latin exemplum"}}

**Start:**
"""

# ==========================================
# 5. UI 布局与主程序 (Main Execution)
# ==========================================

st.title("⚡️ Vocab Flow Ultra")

# 检查 CSV 文件
if not VOCAB_DICT:
    st.warning("⚠️ 未找到词频表文件！请将 `coca_cleaned.csv` 放入根目录，否则词频筛选功能将失效。")

# 标签页
tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

# --- Tab 1: 指南 ---
with tab_guide:
    st.markdown("""
    <div class="guide-step">
    <span class="guide-title">步骤 1: 提取 (Extract)</span>
    上传文档 (PDF, DOCX, EPUB, TXT) 或粘贴文本。系统会自动清洗文本，还原词形 (<i>went -> go</i>)，并根据 COCA 词频表进行筛选。
    </div>
    
    <div class="guide-step">
    <span class="guide-title">步骤 2: 生成 Prompts (Generate)</span>
    系统会将单词分组。复制生成的 Prompt 发送给 AI (ChatGPT, Claude 等)。
    </div>
    
    <div class="guide-step">
    <span class="guide-title">步骤 3: 制作 Anki (Create)</span>
    将 AI 返回的 JSON 粘贴回 "Anki 制作" 标签页，即可生成 <code>.apkg</code> 文件。
    </div>
    """, unsafe_allow_html=True)

# --- Tab 2: 提取 ---
with tab_extract:
    col1, col2 = st.columns(2)
    with col1:
        # 默认 8000，步长 500
        min_r = st.number_input("忽略排名前 N 的词 (Min Rank)", min_value=1, max_value=20000, value=8000, step=500, help="排名高于此（如 the, is）的常用词将被忽略。")
    with col2:
        # 默认 15000，步长 500
        max_r = st.number_input("忽略排名后 N 的词 (Max Rank)", min_value=1, max_value=50000, value=15000, step=500, help="排名低于此的生僻词将被忽略。")
    
    include_unknown = st.checkbox("🔓 包含生僻词/人名 (Rank > 20000)", value=False)
    
    # 文件输入
    uploaded_file = st.file_uploader("📂 上传文件", type=['txt', 'pdf', 'docx', 'epub'], key=st.session_state['uploader_id'])
    pasted_text = st.text_area("📄 或粘贴文本", height=100, key="paste_key")
    
    # 按钮
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        clear_btn = st.button("🗑️ 清空", on_click=clear_all_state)
    with c_btn2:
        analyze_btn = st.button("🚀 开始分析与提取", type="primary")

    if analyze_btn:
        text_content = ""
        if uploaded_file:
            with st.spinner("正在读取文件..."):
                text_content = extract_text_from_file(uploaded_file)
        elif pasted_text:
            text_content = pasted_text
        
        if len(text_content.strip()) > 5:
            start_time = time.time()
            with st.status("正在处理 NLP...", expanded=True) as status:
                status.write("🔍 分词与词形还原中...")
                data, raw_count = analyze_logic(text_content, min_r, max_r, include_unknown)
                status.write(f"✅ 找到 {len(data)} 个生词。")
                
                st.session_state['gen_words_data'] = data
                st.session_state['raw_count'] = raw_count
                st.session_state['process_time'] = time.time() - start_time
                status.update(label="分析完成", state="complete", expanded=False)
        else:
            st.error("⚠️ 请提供有效文本或文件。")

    # 结果显示
    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        # 指标
        m1, m2, m3 = st.columns(3)
        m1.metric("原文总词数", f"{st.session_state['raw_count']:,}")
        m2.metric("目标生词数", f"{len(words_only)}")
        m3.metric("耗时", f"{st.session_state['process_time']:.2f}s")
        
        # 预览
        with st.expander("📋 生词列表预览", expanded=False):
            show_rank = st.toggle("显示排名 (Rank)")
            preview_str = ", ".join([f"{w} ({r})" if show_rank else w for w, r in data_pairs])
            st.markdown(f'<div class="scrollable-text">{preview_str}</div>', unsafe_allow_html=True)
            st.button("📋 复制列表到剪贴板", on_click=lambda: st.write(st.clipboard(preview_str)) or st.toast("已复制！"))

        st.markdown("### ⚙️ Prompt 设置")
        
        # Prompt 配置
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            front_mode = st.selectbox("正面内容", ["单词 (Word)", "短语/搭配 (Phrase)"])
        with pc2:
            def_mode = st.selectbox("释义语言", ["英文", "中文", "中英双语"])
        with pc3:
            # 默认100，最大150，最小1，步长1
            batch_size = st.number_input("AI 分组大小 (Batch Size)", min_value=1, max_value=150, value=100, step=1)
            
        ex_count = st.slider("例句数量", 1, 3, 1)
        need_ety = st.checkbox("包含词源/词根", value=True)
        
        # 生成批次
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        st.info(f"已生成 {len(batches)} 组 Prompt。")
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📝 Prompt 第 {idx+1} 组 (共 {len(batch)} 词)"):
                prompt = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.code(prompt, language="text")

# --- Tab 3: Anki 制作 ---
with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    
    st.info("请将 AI 的 JSON 回复粘贴到此处。支持连续粘贴多次回复。")
    
    if 'anki_input_text' not in st.session_state:
        st.session_state['anki_input_text'] = ""
        
    ai_resp = st.text_area("JSON 输入框", height=200, key="anki_input_text")
    deck_name = st.text_input("牌组名称", f"Vocab_{get_beijing_time_str()}")
    
    if st.button("🛠️ 生成 .apkg 文件", type="primary"):
        if ai_resp.strip():
            parsed_data = parse_anki_data(ai_resp)
            if parsed_data:
                # 预览表格
                df_view = pd.DataFrame(parsed_data)
                st.write(f"✅ 成功解析 {len(parsed_data)} 张卡片。")
                st.dataframe(df_view, use_container_width=True, hide_index=True)
                
                # 生成文件
                f_path = generate_anki_package(parsed_data, deck_name)
                
                # 下载按钮
                with open(f_path, "rb") as f:
                    st.download_button(
                        label=f"📥 下载 {deck_name}.apkg",
                        data=f,
                        file_name=f"{deck_name}.apkg",
                        mime="application/octet-stream"
                    )
            else:
                st.error("❌ 未找到有效的 JSON 数据，请检查格式。")
        else:
            st.warning("⚠️ 输入为空。")