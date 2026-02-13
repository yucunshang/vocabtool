import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache

UNKNOWN_RANK = 99999
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
REPEAT_CHAR_PATTERN = re.compile(r'(.)\1{2,}')
VOWEL_PATTERN = re.compile(r'[aeiouy]')
CODE_FENCE_LANG_PATTERN = re.compile(r'```[a-zA-Z]*\n?')
CODE_FENCE_PATTERN = re.compile(r'```')

# ==========================================
# 0. 濡炪倗鏁诲浼存煀瀹ュ洨鏋?
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="馃摌",
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 闁告柣鍔嶉埀?Key 闁告帗绻傞～鎰板礌?
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
    
    /* 婵犲﹥鑹炬慨鈺冣偓鍦嚀濞呮帡寮藉畡鎵 */
    .scrollable-text {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        background-color: #fafafa;
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    /* 闁圭娲ゅ畷锟犲冀瀹勬壆纭€ (濮掓稒顭堥璇裁归崨鏉款棌) */
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .guide-title { font-size: 18px; font-weight: bold; color: #0f172a; margin-bottom: 10px; display: block; }
    .guide-tip { font-size: 14px; color: #64748b; background: #eef2ff; padding: 8px; border-radius: 4px; margin-top: 8px; }

    /* 闁圭娲ゅ畷锟犲冀瀹勬壆纭€ (濠㈣埖绮撳Λ鍨熼垾宕囩闂侇偄鍊块崢? */
    @media (prefers-color-scheme: dark) {
        .guide-step { background-color: #262730; border-left: 5px solid #4da6ff; box-shadow: none; border: 1px solid #3d3d3d; border-left: 5px solid #4da6ff; }
        .guide-title { color: #e0e0e0; }
        .guide-tip { background-color: #31333F; color: #b0b0b0; border: 1px solid #444; }
        .scrollable-text { background-color: #262730; border: 1px solid #444; color: #ccc; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 閻犙冨缁噣骞婇幒鎴濐潱閺?
# ==========================================
@st.cache_resource(show_spinner="婵繐绲藉﹢顏堝礉閻樼儤绁?NLP 鐎殿喗娲橀幖?..")
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

@st.cache_resource
def get_file_parsers():
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

@st.cache_resource
def get_genanki():
    import genanki
    return genanki

@st.cache_data
def load_vocab_data():
    """
    闁告梻濮惧ù?COCA 閻犲洤绉归。鍓佹偘?
    """
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            df = df.dropna(subset=[w_col])
            # 缂備胶鍠嶇粩瀛樻姜椤掆偓閻剟宕樺▎娆戠闁告宕甸埞鏍冀?
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
    """
    鐎殿喖鎼慨蹇撱€掗崨顖楁晞闁?
    闂傚嫨鍊撶花鈥炽€掗崨瀛樼彑闁告帒妫欓悗鐣岀磼閹惧浜柨娑樼焷缁绘洘瀵煎璺烘缂傚喚鍠楅弸鍐╃閺堢數鐟愬ù鑲╁Т濞呮帡宕仦鐐€柡鍫墲缁额參宕楅妷锔绘敱
    """
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. 闁哄秶顭堢缓楣冩焻閺勫繒甯?(濞村吋锚鐎垫煡鎮?
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            for encoding in ['utf-8', 'gb18030', 'latin-1']:
                try:
                    text = bytes_data.decode(encoding)
                    break
                except: continue
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            page_texts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
            text = "\n".join(page_texts)
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'epub':
            tmp_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                book = epub.read_epub(tmp_path)
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text += soup.get_text(separator=' ', strip=True) + " "
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        return f"Error: {e}"
    return text

def is_valid_word(word):
    """
    闁搞劌鍟┃鍥╂嫚瀹ュ棛顏告繛?
    """
    if len(word) < 2: return False
    if len(word) > 25: return False 
    if REPEAT_CHAR_PATTERN.search(word): return False
    if not VOWEL_PATTERN.search(word): return False
    return True

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    """
    V31 濞村吋锚鐎佃尙绮诲Δ浣恒€婇柨?
    1. 缂備胶鍠曢鎼佹⒓閸涢偊鍤㈤悷鏇炴濞插﹪鎮?(Reading Coverage)
    2. 闁圭粯鍔曡ぐ鍥儎椤旂晫鍨奸柣銏㈠枙閻?(Target Extraction)
    """
    _, lemminflect = load_nlp_resources()
    
    @lru_cache(maxsize=20000)
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    # 1. 閻庣濮ゅ妤呭礆閸℃氨妲?
    raw_tokens = TOKEN_PATTERN.findall(text)
    total_raw_count = len(raw_tokens)
    
    # 2. 缂備胶鍠曢鍝ユ嫚瀹ュ鏆?
    valid_tokens = []
    for t in raw_tokens:
        tl = t.lower()
        if is_valid_word(tl):
            valid_tokens.append(tl)
    token_counts = Counter(valid_tokens)
    
    stats_known_count = 0  
    stats_target_count = 0 
    stats_valid_total = sum(token_counts.values()) 
    
    final_candidates = [] 
    seen_lemmas = set()
    
    # 3. 闂侇剙绉村?
    for w, count in token_counts.items():
        # A. 閻犱緤绱曢悾?Lemma
        lemma = get_lemma_local(w)
        
        # B. 闁兼儳鍢茶ぐ?Rank
        rank_lemma = VOCAB_DICT.get(lemma, UNKNOWN_RANK)
        rank_orig = VOCAB_DICT.get(w, UNKNOWN_RANK)
        
        if rank_lemma != UNKNOWN_RANK and rank_orig != UNKNOWN_RANK:
            best_rank = min(rank_lemma, rank_orig)
        elif rank_lemma != UNKNOWN_RANK:
            best_rank = rank_lemma
        else:
            best_rank = rank_orig
            
        # --- 缂備胶鍠曢鎼佹焻閺勫繒甯?---
        if best_rank < current_lvl:
            stats_known_count += count
        elif current_lvl <= best_rank <= target_lvl:
            stats_target_count += count
            
        # --- 闁圭粯鍔曡ぐ鍥焻閺勫繒甯?---
        is_in_range = (best_rank >= current_lvl and best_rank <= target_lvl)
        is_unknown_included = (best_rank == UNKNOWN_RANK and include_unknown)
        
        if is_in_range or is_unknown_included:
            word_to_keep = lemma if rank_lemma != UNKNOWN_RANK else w
            
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    # 闁圭儤甯掔花?
    final_candidates.sort(key=lambda x: x[1])
    
    # 閻犱緤绱曢悾濠氭儌閹冪€绘慨?
    coverage_ratio = (stats_known_count / stats_valid_total) if stats_valid_total > 0 else 0
    target_ratio = (stats_target_count / stats_valid_total) if stats_valid_total > 0 else 0
    
    stats_info = {
        "coverage": coverage_ratio,
        "target_density": target_ratio
    }
    
    return final_candidates, total_raw_count, stats_info

# ==========================================
# (濞村吋锚鐎垫煡鎮? JSON 閻熸瑱绲鹃悗浠嬫焻閺勫繒甯?
# ==========================================
def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.strip()
    text = CODE_FENCE_LANG_PATTERN.sub('', text)
    text = CODE_FENCE_PATTERN.sub('', text).strip()
    
    json_objects = []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            json_objects = data
    except:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and (text[pos].isspace() or text[pos] == ','):
                pos += 1
            if pos >= len(text):
                break
            try:
                obj, index = decoder.raw_decode(text[pos:])
                json_objects.append(obj)
                pos += index
            except:
                pos += 1

    seen_phrases_lower = set()

    for data in json_objects:
        if not isinstance(data, dict):
            continue

        lower_key_map = {k.lower(): v for k, v in data.items()}
        missing = object()
            
        def get_val(keys_list):
            for k in keys_list:
                if k in data:
                    return data[k]
                val = lower_key_map.get(k.lower(), missing)
                if val is not missing:
                    return val
            return ""

        front_text = get_val(['w', 'word', 'phrase', 'term'])
        meaning = get_val(['m', 'meaning', 'def', 'definition'])
        examples = get_val(['e', 'example', 'examples', 'sentence'])
        etymology = get_val(['r', 'root', 'etymology', 'origin'])

        if not front_text or not meaning:
            continue
        
        front_text = str(front_text).replace('**', '').strip()
        meaning = str(meaning).strip()
        examples = str(examples).strip()
        etymology = str(etymology).strip()
        
        if etymology.lower() in ["none", "null", ""]:
            etymology = ""

        if front_text.lower() in seen_phrases_lower: 
            continue
        seen_phrases_lower.add(front_text.lower())

        parsed_cards.append({
            'front_phrase': front_text,
            'meaning': meaning,
            'examples': examples,
            'etymology': etymology
        })

    return parsed_cards

# ==========================================
# 3. Anki 闁汇垻鍠愰崹?(濞村吋锚鐎? 閻庢稒銇炵紞瀣磼閻斿墎顏?
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki = get_genanki()
    
    # 濞村吋锚鐎?CSS: 濞撴艾顑呰ぐ鐐哄椽瀹€鍐Г婵犙勫姇閻⊙勬媴閹炬剚鏉诲鍫嗗啫鐓?20px
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; line-height: 1.3; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .nightMode .definition { color: #e0e0e0; }
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 20px; line-height: 1.5; margin-bottom: 15px; text-align: left; }
    .nightMode .examples { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .footer-info { margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 10px; text-align: left; }
    .etymology { display: block; font-size: 20px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; line-height: 1.4; border: 1px solid #fef3c7; }
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
            <div class="footer-info"><div class="etymology">妫ｅ啫鍟?<b>閻犲洤绉电花?</b> {{Etymology}}</div></div>
            {{/Etymology}}
            ''',
        }], css=CSS
    )
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    
    for c in cards_data:
        f_phrase = str(c.get('front_phrase', ''))
        f_meaning = str(c.get('meaning', ''))
        f_examples = str(c.get('examples', '')).replace('\n','<br>')
        f_etymology = str(c.get('etymology', ''))
        
        deck.add_note(genanki.Note(
            model=model, 
            fields=[f_phrase, f_meaning, f_examples, f_etymology]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def build_cards_signature(cards):
    try:
        return json.dumps(cards, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(cards)

# ==========================================
# 4. Prompt Logic (濞村吋锚鐎? 閻犲浂鍘虹粻鐔煎储閻旈鎽嶉柟?
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)

    if front_mode == "单词 (Word)":
        w_instr = "Key `w`: The word itself (lowercase)."
    else:
        w_instr = "Key `w`: A short practical collocation/phrase (2-5 words) that naturally contains the word."

    if def_mode == "中文":
        m_instr = "Key `m`: Concise Chinese definition of the **word** (max 10 chars). NOT the definition of the phrase."
    elif def_mode == "中英双语":
        m_instr = "Key `m`: English definition + Chinese definition of the **word**."
    else:
        m_instr = "Key `m`: English definition of the **word** (concise)."

    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."
    r_instr = (
        "Key `r`: Simplified Chinese Etymology (Root/Prefix) corresponding to this specific meaning."
        if need_ety else
        "Key `r`: Leave this empty string \"\"."
    )

    return f"""
Task: Create Anki cards.
Words: {w_list}

**CRITICAL: SEMANTIC ATOMICITY**
1. **Consistency**: The Word/Phrase (`w`), Meaning (`m`), Example (`e`), and Etymology (`r`) MUST all correspond to the **same specific context/meaning**.
2. **No Mixing**: Do NOT mix definitions. (e.g., If `w` is "bracket" in a tax context, `m` must be "grade/category", `e` must be about taxes. Do NOT give the definition of "punctuation mark").
3. **Definition Focus**: Even if `w` is a phrase (e.g. "give up"), `m` should explain the core meaning derived from it.

**Output Format: NDJSON (One line per object).**

**Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Keys:** `w` (Front), `m` (Meaning), `e` (Examples), `r` (Etymology)

**Example (Correct Consistency):**
{{"w": "bracket", "m": "绛夌骇/妗ｆ", "e": "He is in the highest income tax bracket.", "r": "from braguette (codpiece)"}}

**Start:**
"""

st.title("Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("缺少 `coca_cleaned.csv`")

tab_guide, tab_extract, tab_anki = st.tabs(["使用指南", "1. 单词提取", "2. Anki 制作"])

with tab_guide:
    st.markdown("""
### Vocab Flow Ultra
这个工具用于从文本中提取目标词汇，并生成可用于 Anki 制卡的 JSON 提示词。

1. 在**单词提取**中上传文件或粘贴文本后开始分析。
2. 复制分组 Prompt，交给 AI 生成 JSON。
3. 在**Anki 制作**粘贴 JSON，生成并下载 `.apkg`。
""")
with tab_extract:
    mode_context, mode_rank = st.tabs(["语境分析", "词频列表"])

    with mode_context:
        st.info("全能模式：自动词形还原、去重、清洗噪声词。")

        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略前 N 名词", 1, 20000, 6000, step=100)
        targ = c2.number_input("保留到 N 名词", 2000, 50000, 10000, step=500)
        include_unknown = st.checkbox("包含生僻词（Rank > 20000）", value=False)

        uploaded_file = st.file_uploader("上传文件（TXT/PDF/DOCX/EPUB）", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("或粘贴文本", height=100, key="paste_key")

        if st.button("开始分析", type="primary"):
            with st.status("处理中...", expanded=True) as status:
                start_time = time.time()
                status.write("读取输入并清洗词元...")
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text

                if len(raw_text) > 2:
                    status.write("执行分析并计算覆盖率...")
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, include_unknown)

                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time

                    status.update(label="分析完成", state="complete", expanded=False)
                else:
                    status.update(label="输入内容过短", state="error")

        st.button("清空", type="secondary", on_click=clear_all_state)

    with mode_rank:
        gen_type = st.radio("模式", ["顺序", "随机"], horizontal=True)
        if "顺序" in gen_type:
            c_a, c_b = st.columns(2)
            s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
            count = c_b.number_input("数量", 10, 500, 50, step=10)
            if st.button("按排名生成"):
                start_time = time.time()
                if FULL_DF is not None:
                    r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                    w_col = next(c for c in FULL_DF.columns if 'word' in c)
                    subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                    data_list = list(zip(subset[w_col], subset[r_col]))
                    st.session_state['gen_words_data'] = data_list
                    st.session_state['raw_count'] = 0
                    st.session_state['stats_info'] = None
                    st.session_state['process_time'] = time.time() - start_time
        else:
            c_min, c_max, c_cnt = st.columns([1, 1, 1])
            min_r = c_min.number_input("最小 Rank", 1, 20000, 1, step=100)
            max_r = c_max.number_input("最大 Rank", 1, 25000, 5000, step=100)
            r_count = c_cnt.number_input("数量", 10, 200, 50, step=10)
            if st.button("随机抽取"):
                start_time = time.time()
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
                        st.session_state['process_time'] = time.time() - start_time

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]

        st.divider()
        st.markdown("### 分析报告")

        k1, k2, k3, k4 = st.columns(4)
        raw_c = st.session_state.get('raw_count', 0)
        stats = st.session_state.get('stats_info', {})

        k1.metric("总词元数", f"{raw_c:,}")

        if stats:
            k2.metric("熟词覆盖率", f"{stats.get('coverage', 0):.1%}")
            k3.metric("目标词密度", f"{stats.get('target_density', 0):.1%}")
        else:
            k2.metric("熟词覆盖率", "--")
            k3.metric("目标词密度", "--")

        k4.metric("提取词数", f"{len(words_only)}")

        show_rank = st.checkbox("显示 Rank", value=False)

        if show_rank:
            display_text = ", ".join([f"{w}[{r}]" for w, r in data_pairs])
        else:
            display_text = ", ".join(words_only)

        with st.expander("全部提取词", expanded=False):
            st.markdown(f'<div class="scrollable-text">{display_text}</div>', unsafe_allow_html=True)
            st.caption("提示：可从文本框复制，或使用代码块右上角复制按钮。")
            st.code(display_text, language="text")

        with st.expander("Prompt 设置", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("正面内容", ["短语搭配 (Phrase)", "单词 (Word)"])
            def_mode = col_s2.selectbox("释义语言", ["英文", "中文", "中英双语"])

            col_s3, col_s4 = st.columns(2)
            ex_count = col_s3.slider("每词例句数", 1, 3, 1)
            need_ety = col_s4.checkbox("包含词源", value=True)

        batch_size = st.number_input("AI 分组大小", 50, 500, 150, step=10)
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]

        for idx, batch in enumerate(batches):
            with st.expander(f"第 {idx + 1} 组（{len(batch)} 词）", expanded=(idx == 0)):
                prompt_text = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.caption("移动端复制区")
                st.text_area(f"text_area_{idx}", value=prompt_text, height=100, label_visibility="collapsed")
                st.caption("桌面端代码块")
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 制作 Anki 牌组")

    if 'anki_cards_cache' not in st.session_state:
        st.session_state['anki_cards_cache'] = None
    if 'anki_package_cache' not in st.session_state:
        st.session_state['anki_package_cache'] = None

    def reset_anki_state():
        st.session_state['anki_cards_cache'] = None
        st.session_state['anki_package_cache'] = None
        if 'anki_input_text' in st.session_state:
            st.session_state['anki_input_text'] = ""

    col_input, _ = st.columns([3, 1])
    with col_input:
        bj_time_str = get_beijing_time_str()
        deck_name = st.text_input("牌组名称", f"Vocab_{bj_time_str}", help="导入 Anki 后显示的名称")

    st.caption("在下方粘贴 AI 生成的 JSON（支持多批次追加粘贴）。")

    ai_resp = st.text_area(
        "JSON 输入",
        height=300,
        key="anki_input_text",
        placeholder='''[
  {"w": "serendipity", "m": "a happy accidental discovery", "e": "It was pure serendipity.", "r": "coined by Horace Walpole"},
  ...
]'''
    )

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        start_gen = st.button("开始生成", type="primary", use_container_width=True)
    with c_btn2:
        st.button("清空重置", type="secondary", on_click=reset_anki_state)

    if start_gen or st.session_state['anki_cards_cache'] is not None:
        if start_gen:
            if not ai_resp.strip():
                st.warning("输入为空，请先粘贴 AI 生成的 JSON。")
            else:
                with st.spinner("正在解析 JSON 并构建卡片..."):
                    parsed_data = parse_anki_data(ai_resp)
                    if parsed_data:
                        st.session_state['anki_cards_cache'] = parsed_data
                        st.session_state['anki_package_cache'] = None
                        st.success(f"已提取 {len(parsed_data)} 张卡片。")
                    else:
                        st.error("解析失败：未找到有效 JSON 对象。")
                        st.session_state['anki_cards_cache'] = None
                        st.session_state['anki_package_cache'] = None

        if st.session_state['anki_cards_cache']:
            cards = st.session_state['anki_cards_cache']

            with st.expander("卡片预览（前 50 条）", expanded=True):
                df_view = pd.DataFrame(cards)
                df_preview = df_view.rename(columns={
                    'front_phrase': '正面',
                    'meaning': '背面',
                    'examples': '例句',
                    'etymology': '词源'
                })
                st.dataframe(df_preview.head(50), use_container_width=True, hide_index=True)

            try:
                cards_signature = build_cards_signature(cards)
                cached_pkg = st.session_state.get('anki_package_cache')
                need_regen = (
                    not cached_pkg
                    or cached_pkg.get('deck_name') != deck_name
                    or cached_pkg.get('cards_signature') != cards_signature
                )

                if need_regen:
                    f_path = generate_anki_package(cards, deck_name)
                    with open(f_path, "rb") as f:
                        file_data = f.read()
                    if os.path.exists(f_path):
                        os.remove(f_path)
                    st.session_state['anki_package_cache'] = {
                        'deck_name': deck_name,
                        'cards_signature': cards_signature,
                        'file_data': file_data
                    }
                else:
                    file_data = cached_pkg['file_data']

                st.download_button(
                    label=f"下载 {deck_name}.apkg",
                    data=file_data,
                    file_name=f"{deck_name}.apkg",
                    mime="application/octet-stream",
                    type="primary",
                    help="下载后导入 Anki"
                )
            except Exception as e:
                st.error(f".apkg 生成失败：{e}")
