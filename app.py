import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
import zlib
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra (Pro)", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    .scrollable-text {
        max-height: 200px; overflow-y: auto; padding: 10px;
        border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;
        font-family: monospace; white-space: pre-wrap;
    }
    .stCodeBlock { border: 1px solid #d1d5db; border-radius: 8px; }
    @media (prefers-color-scheme: dark) {
        .scrollable-text { background-color: #262730; border: 1px solid #444; color: #ccc; }
        .stCodeBlock { border: 1px solid #444; }
    }
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
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. 文本提取逻辑 (无时间过滤版)
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    _, tempfile = get_genanki() 
    
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            for encoding in ['utf-8', 'gb18030', 'latin-1']:
                try: text = bytes_data.decode(encoding); break
                except: continue
        
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        
        elif file_type == 'epub':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator=' ', strip=True) + " "
            os.remove(tmp_path)
            
        # ==========================================
        # Kindle DB 逻辑 (提取全部)
        # ==========================================
        elif file_type in ['db', 'sqlite']:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
                tmp_db.write(uploaded_file.getvalue())
                tmp_db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(tmp_db_path)
                cursor = conn.cursor()
                
                try:
                    # 优先提取 STEM (词干)
                    cursor.execute("SELECT stem FROM WORDS WHERE stem IS NOT NULL")
                    rows = cursor.fetchall()
                    text = " ".join([r[0] for r in rows if r[0]])
                    
                    # 兜底：如果 STEM 为空，提取原始 Word
                    if not text:
                         cursor.execute("SELECT word FROM WORDS")
                         rows = cursor.fetchall()
                         text = " ".join([r[0] for r in rows if r[0]])

                except Exception as db_err:
                    text = f"Error reading DB schema: {db_err}"
                
                conn.close()
            except Exception as e:
                text = f"Error connecting to DB: {e}"
            finally:
                if os.path.exists(tmp_db_path): os.remove(tmp_db_path)

    except Exception as e:
        return f"Error: {e}"
    
    return text

def is_valid_word(word):
    if len(word) < 2: return False
    if len(word) > 25: return False 
    if re.search(r'(.)\1{2,}', word): return False
    if not re.search(r'[aeiouy]', word): return False
    return True

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    nltk, lemminflect = load_nlp_resources()
    
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_raw_count = len(raw_tokens)
    
    valid_tokens = [t.lower() for t in raw_tokens if is_valid_word(t.lower())]
    token_counts = Counter(valid_tokens)
    
    stats_known_count = 0  
    stats_target_count = 0 
    stats_valid_total = sum(token_counts.values()) 
    
    final_candidates = [] 
    seen_lemmas = set()
    
    for w, count in token_counts.items():
        lemma = get_lemma_local(w)
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        if rank_lemma != 99999 and rank_orig != 99999:
            best_rank = min(rank_lemma, rank_orig)
        elif rank_lemma != 99999:
            best_rank = rank_lemma
        else:
            best_rank = rank_orig
            
        if best_rank < current_lvl:
            stats_known_count += count
        elif current_lvl <= best_rank <= target_lvl:
            stats_target_count += count
            
        is_in_range = (best_rank >= current_lvl and best_rank <= target_lvl)
        is_unknown_included = (best_rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            word_to_keep = lemma if rank_lemma != 99999 else w
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    final_candidates.sort(key=lambda x: x[1])
    
    coverage_ratio = (stats_known_count / stats_valid_total) if stats_valid_total > 0 else 0
    target_ratio = (stats_target_count / stats_valid_total) if stats_valid_total > 0 else 0
    
    stats_info = {
        "coverage": coverage_ratio,
        "target_density": target_ratio
    }
    return final_candidates, total_raw_count, stats_info

# ==========================================
# 数据解析逻辑
# ==========================================
def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.strip()
    
    code_block = re.search(r'```(?:text|csv)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1)
    else:
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)
    
    lines = text.split('\n')
    seen_phrases = set()

    for line in lines:
        line = line.strip()
        if not line or "|||" not in line: 
            continue
            
        parts = line.split("|||")
        if len(parts) < 2: 
            continue
        
        w = parts[0].strip()
        m = parts[1].strip()
        e = parts[2].strip() if len(parts) > 2 else ""
        r = parts[3].strip() if len(parts) > 3 else ""

        if w.lower() in seen_phrases: 
            continue
        seen_phrases.add(w.lower())
        
        parsed_cards.append({
            'w': w, 'm': m, 'e': e, 'r': r
        })

    return parsed_cards

# ==========================================
# Anki 生成逻辑
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
    
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .meaning { font-size: 20px; font-weight: bold; color: #222; margin-bottom: 15px; text-align: left; }
    .nightMode .meaning { color: #e0e0e0; }
    .example { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 18px; text-align: left; margin-bottom: 15px; }
    .nightMode .example { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; border: 1px solid #fef3c7; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    
    MODEL_ID = 1842957301 
    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))

    model = genanki.Model(
        MODEL_ID, 
        'VocabFlow Phrase Model',
        fields=[
            {'name': 'Phrase'}, {'name': 'Meaning'},
            {'name': 'Example'}, {'name': 'Etymology'}
        ],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{Phrase}}</div>', 
            'afmt': '''
            {{FrontSide}}
            <hr>
            <div class="meaning">{{Meaning}}</div>
            <div class="example">🗣️ {{Example}}</div>
            {{#Etymology}}
            <div class="etymology">🌱 词源: {{Etymology}}</div>
            {{/Etymology}}
            ''',
        }], css=CSS
    )
    
    deck = genanki.Deck(DECK_ID, deck_name)
    
    for c in cards_data:
        deck.add_note(genanki.Note(
            model=model, 
            fields=[
                str(c.get('w', '')), str(c.get('m', '')), 
                str(c.get('e', '')), str(c.get('r', ''))
            ]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# Prompt 逻辑
# ==========================================
def get_ai_prompt(words):
    w_list = ", ".join(words)
    return f"""
# Role
You are an expert English Lexicographer and Anki Card Designer. Your goal is to convert a list of target words into high-quality, import-ready Anki flashcards focusing on **natural collocations** (word chunks).

# Input Data
{w_list}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
   `Natural Phrase/Collocation` ||| `Concise Definition of the Phrase` ||| `Short Example Sentence` ||| `Etymology breakdown (Simplified Chinese)`

# Field Constraints (Strict)
1. **Field 1: Phrase (CRITICAL)**
   - DO NOT output the single target word.
   - You MUST generate a high-frequency **collocation** or **short phrase** containing the target word.
   - Example: If input is "rain", output "heavy rain" or "torrential rain".
   
2. **Field 2: Definition (English)**
   - Define the *phrase*, not just the isolated word. Keep it concise (B2-C1 level English).

3. **Field 3: Example**
   - A short, authentic sentence containing the phrase.

4. **Field 4: Roots/Etymology (Simplified Chinese)**
   - Format: `prefix- (meaning) + root (meaning) + -suffix (meaning)`.
   - If no classical roots exist, explain the origin briefly in Chinese.
   - Use Simplified Chinese for meanings.

# Valid Example (Follow this logic strictly)
Input: altruism
Output:
motivated by altruism ||| acting out of selfless concern for the well-being of others ||| His donation was motivated by altruism, not a desire for fame. ||| alter (其他) + -ism (主义/行为)

Input: hectic
Output:
a hectic schedule ||| a timeline full of frantic activity and very busy ||| She has a hectic schedule with meetings all day. ||| hect- (持续的/习惯性的 - 来自希腊语hektikos) + -ic (形容词后缀)

# Task
Process the provided input list strictly adhering to the format above.
"""

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv` 文件，请检查目录。")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ 卡片制作"])

with tab_guide:
    st.markdown("""
    ### 👋 欢迎使用 Vocab Flow Ultra
    
    本工具旨在将您的阅读积累转化为永久记忆。支持从各种文档或生词本中提取单词，并利用 AI 生成高质量 Anki 卡片。
    
    #### 📂 全面支持的文件格式
    | 类型 | 扩展名 | 说明 |
    | :--- | :--- | :--- |
    | **Kindle 生词本** | `.db` / `.sqlite` | 直接上传 `system/vocabulary/vocab.db`。 |
    | **电子书** | `.epub` | 自动解析章节内容，去除 HTML 标签。 |
    | **文档** | `.pdf` | 支持扫描版以外的标准 PDF 文本提取。 |
    | **Word** | `.docx` | 提取段落文本，忽略图片和表格。 |
    | **纯文本** | `.txt` | 支持 UTF-8, GBK 等常见编码。 |

    ---

    #### 💡 Kindle 生词本：防卡死技巧
    如果您习惯删除 `vocab.db` 来清空生词本，请务必执行以下操作，否则 Kindle 将无法记录新词：
    
    1.  **删除文件**：将 `vocab.db` 移出或删除。
    2.  **必须重启**：长按电源键 **40秒**，或在 Kindle 搜索栏输入 `;restart` 并回车。
    3.  **恢复正常**：重启后系统会自动重建数据库，生词本即可恢复使用。
    
    ---
    
    #### ⚡️ 极速工作流
    1. **提取**：在“单词提取”页上传文件。
    2. **生成**：点击“复制代码”发送给 AI (ChatGPT/Claude)。
    3. **制作**：将 AI 返回的代码块粘贴回“卡片制作”页，生成 `.apkg` 包。
    """)

with tab_extract:
    mode_context, mode_direct, mode_rank = st.tabs(["📄 语境分析", "📝 直接输入", "🔢 词频列表"])
    
    # 模式1：语境分析
    with mode_context:
        st.info("💡 **智能模式**：自动进行词形还原、去重和垃圾词清洗。")
        
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100)
        targ = c2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500)
        include_unknown = st.checkbox("🔓 包含生僻词/人名 (Rank > 20000)", value=False)

        uploaded_file = st.file_uploader(
            "📂 上传文件 (支持 .db, .pdf, .docx, .epub, .txt)", 
            key=st.session_state['uploader_id']
        )
        
        # === 简洁的 Kindle 提示 ===
        with st.expander("❓ 删除了 vocab.db 导致无法记录生词？"):
            st.warning("**解决方法：必须重启 Kindle**\n\n请长按电源键 40 秒，或者在搜索栏输入 `;restart` 并回车。")
        
        pasted_text = st.text_area("📄 ...或在此粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理中...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 正在读取文件...")
                
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 2:
                    status.write("🔍 正在分析文本复杂度...")
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, include_unknown)
                    
                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time
                    
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容为空或太短", state="error")
    
    # 模式2：直接输入
    with mode_direct:
        st.info("💡 **直接模式**：不进行词频过滤，直接为粘贴的单词生成 Prompt。")
        raw_input = st.text_area("✍️ 粘贴单词列表 (每行一个 或 逗号分隔)", height=200, placeholder="altruism\nhectic\nserendipity")
        
        if st.button("🚀 生成列表", key="btn_direct", type="primary"):
            if raw_input.strip():
                words = [w.strip() for w in re.split(r'[,\n\t]+', raw_input) if w.strip()]
                seen = set()
                unique_words = []
                for w in words:
                    if w.lower() not in seen:
                        seen.add(w.lower())
                        unique_words.append(w)
                
                data_list = []
                for w in unique_words:
                    rank = VOCAB_DICT.get(w.lower(), 99999) 
                    data_list.append((w, rank))
                
                st.session_state['gen_words_data'] = data_list
                st.session_state['raw_count'] = len(unique_words)
                st.session_state['stats_info'] = None 
                
                st.success(f"✅ 已加载 {len(unique_words)} 个单词")
            else:
                st.warning("⚠️ 内容为空，请先粘贴单词。")

    # 模式3：词频列表
    with mode_rank:
        gen_type = st.radio("生成模式", ["🔢 顺序生成", "🔀 随机抽取"], horizontal=True)
        if "顺序生成" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
             count = c_b.number_input("数量", 10, 5000, 50, step=50)
             if st.button("🚀 生成列表"):
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                     data_list = list(zip(subset[w_col], subset[r_col]))
                     st.session_state['gen_words_data'] = data_list
                     st.session_state['raw_count'] = 0
                     st.session_state['stats_info'] = None
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("最小排名", 1, 20000, 1, step=100)
             max_r = c_max.number_input("最大排名", 1, 25000, 5000, step=100)
             r_count = c_cnt.number_input("抽取数量", 10, 5000, 50, step=50)
             if st.button("🎲 随机抽取"):
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                     candidates = FULL_DF[mask]
                     if len(candidates) > 0:
                         subset = candidates.sample(n=min(r_count, len(candidates))).sort_values(r_col)
                         data_list = list(zip(subset[w_col], subset[r_col]))
                         st.session_state['gen_words_data'] = data_list
                         st.session_state['raw_count'] = 0
                         st.session_state['stats_info'] = None
    
    # 清空按钮
    if st.button("🗑️ 清空重置", type="secondary", on_click=clear_all_state, key="btn_clear_extract"): pass

    # ==========================================
    # 结果展示区
    # ==========================================
    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        st.markdown("### 📊 分析报告")
        
        k1, k2, k3, k4 = st.columns(4)
        raw_c = st.session_state.get('raw_count', 0)
        stats = st.session_state.get('stats_info', {})
        
        k1.metric("总词数", f"{raw_c:,}")
        if stats:
            k2.metric("熟词覆盖率", f"{stats.get('coverage', 0):.1%}")
            k3.metric("生词密度", f"{stats.get('target_density', 0):.1%}")
        else:
            k2.metric("熟词覆盖率", "--")
            k3.metric("生词密度", "--")
        k4.metric("提取生词数", f"{len(words_only)}")
        
        show_rank = st.checkbox("显示单词排名 (Rank)", value=False)
        display_text = ", ".join([f"{w}[{r}]" for w, r in data_pairs]) if show_rank else ", ".join(words_only)
            
        with st.expander("📋 **预览所有单词**", expanded=False):
            st.markdown(f'<div class="scrollable-text">{display_text}</div>', unsafe_allow_html=True)
            st.code(display_text, language="text")

        st.divider()
        st.subheader("🤖 AI 提示词 (一键复制)")
        st.caption("提示：使用最新专家级 Prompt，专注于生成自然搭配和精准词源。")
        
        batch_size = st.number_input("AI 分组大小 (Batch Size)", 50, 500, 100, step=50)
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 第 {idx+1} 组 (共 {len(batch)} 词)", expanded=(idx==0)):
                prompt_text = get_ai_prompt(batch)
                st.markdown("👇 **点击右上角图标复制**")
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    
    if 'anki_cards_cache' not in st.session_state:
        st.session_state['anki_cards_cache'] = None
    
    def reset_anki_state():
        st.session_state['anki_cards_cache'] = None
        if 'anki_input_text' in st.session_state:
             st.session_state['anki_input_text'] = ""

    col_input, col_act = st.columns([3, 1])
    with col_input:
        bj_time_str = get_beijing_time_str()
        deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{bj_time_str}")
    
    st.caption("👇 **在此粘贴 AI 返回的内容 (包含 ```text 代码块也没问题)：**")
    
    ai_resp = st.text_area(
        "输入框", 
        height=300, 
        key="anki_input_text",
        placeholder='```text\nmotivated by altruism ||| acting out of... ||| ...\n```'
    )

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        start_gen = st.button("🚀 生成卡片", type="primary", use_container_width=True)
    with c_btn2:
        st.button("🗑️ 清空重置", type="secondary", on_click=reset_anki_state, key="btn_clear_anki")

    if start_gen or st.session_state['anki_cards_cache'] is not None:
        if start_gen:
            if not ai_resp.strip():
                st.warning("⚠️ 输入框为空，请先粘贴内容。")
            else:
                with st.spinner("正在解析数据..."):
                    parsed_data = parse_anki_data(ai_resp)
                    if parsed_data:
                        st.session_state['anki_cards_cache'] = parsed_data
                        st.success(f"✅ 成功解析 {len(parsed_data)} 张卡片！")
                    else:
                        st.error("❌ 解析失败。未检测到有效内容，请检查分隔符是否为 '|||'")
                        st.session_state['anki_cards_cache'] = None

        if st.session_state['anki_cards_cache']:
            cards = st.session_state['anki_cards_cache']
            
            with st.expander("👀 预览卡片 (前 50 张)", expanded=True):
                df_view = pd.DataFrame(cards)
                df_view.columns = ["正面(短语)", "英文释义", "英文例句", "中文词源"]
                st.dataframe(df_view, use_container_width=True, hide_index=True)

            try:
                f_path = generate_anki_package(cards, deck_name)
                with open(f_path, "rb") as f:
                    file_data = f.read()
                    
                st.download_button(
                    label=f"📥 下载 {deck_name}.apkg",
                    data=file_data,
                    file_name=f"{deck_name}.apkg",
                    mime="application/octet-stream",
                    type="primary"
                )
            except Exception as e:
                st.error(f"生成文件出错: {e}")