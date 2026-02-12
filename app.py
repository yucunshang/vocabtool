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
# 0. Page Configuration & CSS
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra",
    page_icon="⚡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
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

# Initialize Session State
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

# ==========================================
# 1. Resource Lazy Loading & Helpers
# ==========================================

@st.cache_resource(show_spinner="正在加载 NLP 引擎 (首次运行较慢)...")
def load_nlp_resources():
    """
    Lazy load NLTK and Lemminflect to improve startup time.
    """
    import nltk
    import lemminflect
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # Required NLTK packages
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
    Load COCA frequency list. Returns a Dict {word: rank} and the full DataFrame.
    Gracefully handles missing files by returning an empty state.
    """
    # Priority list for filenames
    possible_files = ["coca_cleaned.csv", "vocab.csv", "data.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Identify columns dynamically
            w_col = next((c for c in df.columns if 'word' in c), None)
            r_col = next((c for c in df.columns if 'rank' in c), None)
            
            if not w_col or not r_col:
                return {}, None

            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            
            # Deduplicate keeping the highest rank (lowest number)
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            
            vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            return vocab_dict, df
        except Exception as e:
            st.error(f"Error reading vocabulary file: {e}")
            return {}, None
    return {}, None

# Load global data once
VOCAB_DICT, FULL_DF = load_vocab_data()

def get_beijing_time_str():
    """Returns formatted timestamp string (Beijing Time)."""
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

def clear_all_state():
    """Hard reset of session state."""
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'anki_input_text']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    # Randomize uploader key to force UI reset
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. Core Logic: Extraction & Analysis
# ==========================================

def extract_text_from_file(uploaded_file):
    """Parses PDF, DOCX, EPUB, TXT."""
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
            # Try common encodings
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
            # Handle EPUB requiring a temp file
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
        return f"Error reading file: {str(e)}"
        
    return text

def is_valid_word(word):
    """Heuristic cleaning to remove garbage."""
    if len(word) < 2: return False
    if len(word) > 25: return False
    # Filter strings with 3+ identical consecutive characters
    if re.search(r'(.)\1{2,}', word): return False
    # Must contain at least one vowel (heuristic for English)
    if not re.search(r'[aeiouy]', word): return False
    # No numbers or symbols allowed inside (except hyphen handled earlier)
    if re.search(r'[0-9_]', word): return False
    return True

def analyze_logic(text, min_rank, max_rank, include_unknown):
    """
    The Core Algorithm: Tokenize -> Lemmatize -> Rank Check -> Dedupe.
    Returns: List of (word, rank) tuples, raw_word_count
    """
    nltk, lemminflect = load_nlp_resources()
    
    # 1. Tokenization (keep internal hyphens like 'well-known')
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(raw_tokens)
    
    # 2. Initial cleaning
    clean_tokens = set([t.lower() for t in raw_tokens if is_valid_word(t.lower())])
    
    final_candidates = [] 
    seen_lemmas = set()
    
    for w in clean_tokens:
        # Get Lemma (e.g., went -> go)
        # We try VERB first as it changes most, fallback to None
        try:
            lemma = lemminflect.getLemma(w, upos='VERB')[0]
        except:
            lemma = w
            
        # Get Ranks
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        # Determine effective rank (best of both)
        best_rank = min(rank_lemma, rank_orig)
        
        # Determine output word (prefer Lemma if it has a valid rank)
        word_to_keep = lemma if rank_lemma != 99999 else w
        
        # Filtering Logic
        is_in_range = (min_rank <= best_rank <= max_rank)
        is_unknown_included = (include_unknown and best_rank == 99999)
        
        if is_in_range or is_unknown_included:
            # Deduplication: Use lemma as key
            # Ensures 'go' and 'went' don't both appear
            dedupe_key = lemma
            
            if dedupe_key not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(dedupe_key)
    
    # Sort: High freq (low rank) -> Low freq -> Unknown
    final_candidates.sort(key=lambda x: x[1])
    
    return final_candidates, total_words

# ==========================================
# 3. Anki Parsing & Generation
# ==========================================

def parse_anki_data(raw_text):
    """
    Extracts JSON objects from messy AI response.
    Input: Text that might contain markdown, text, and multiple JSON objects.
    Output: List of dictionaries.
    """
    parsed_cards = []
    # Remove markdown code blocks
    text = raw_text.replace("```json", "").replace("```", "").strip()
    
    # Regex to find JSON-like structures { ... }
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases = set()

    for match in matches:
        json_str = match.group()
        try:
            data = json.loads(json_str, strict=False)
            
            # Extract fields with safe defaults
            front = str(data.get("w", "")).strip()
            meaning = str(data.get("m", "")).strip()
            examples = str(data.get("e", "")).strip()
            etymology = str(data.get("r", "")).strip()
            
            if etymology.lower() in ["none", "", "null"]:
                etymology = ""

            # Basic Validation
            if not front or not meaning:
                continue
            
            # Remove Markdown bolding from front
            front = front.replace('**', '')
            
            # Deduplicate inside this batch
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
    """Generates .apkg file using genanki."""
    import genanki
    import tempfile
    
    # CSS Styling for cards
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
    
    # Create unique Model ID
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
# 4. Prompt Engineering
# ==========================================

def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    # Configurable Instructions
    w_instr = "Key `w`: The word itself (lemma)." if "Word" in front_mode else "Key `w`: A common short phrase/collocation using the word."
    
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
# 5. UI Layout & Main Execution
# ==========================================

st.title("⚡️ Vocab Flow Ultra")

# CSV Check
if not VOCAB_DICT:
    st.warning("⚠️ Dictionary file not found! Please place `coca_cleaned.csv` in the root directory. Rank filtering will not work correctly.")

# Tabs
tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

# --- Tab 1: Guide ---
with tab_guide:
    st.markdown("""
    <div class="guide-step">
    <span class="guide-title">Step 1: Extract</span>
    Upload a document (PDF, DOCX, EPUB, TXT) or paste text. The system cleans the text, lemmatizes words (<i>went -> go</i>), and filters them based on COCA frequency ranking.
    </div>
    
    <div class="guide-step">
    <span class="guide-title">Step 2: Generate Prompts</span>
    The system groups words into batches. Copy the generated Prompt and send it to AI (ChatGPT, Claude, etc.).
    </div>
    
    <div class="guide-step">
    <span class="guide-title">Step 3: Create Anki</span>
    Paste the JSON response from the AI back into the "Anki 制作" tab to generate your <code>.apkg</code> file.
    </div>
    """, unsafe_allow_html=True)

# --- Tab 2: Extraction ---
with tab_extract:
    col1, col2 = st.columns(2)
    with col1:
        min_r = st.number_input("Min Rank (Filter easy words)", 1, 20000, 2000, step=100, help="Words ranked higher than this (e.g., 'the', 'is') will be ignored.")
    with col2:
        max_r = st.number_input("Max Rank (Filter rare words)", 1000, 50000, 15000, step=500, help="Words ranked lower than this will be ignored.")
    
    include_unknown = st.checkbox("🔓 Include Unknown/Rare Words (Rank > 20000)", value=False)
    
    # File Input
    uploaded_file = st.file_uploader("📂 Upload File", type=['txt', 'pdf', 'docx', 'epub'], key=st.session_state['uploader_id'])
    pasted_text = st.text_area("📄 Or Paste Text", height=100, key="paste_key")
    
    # Action Buttons
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        clear_btn = st.button("🗑️ Clear", on_click=clear_all_state)
    with c_btn2:
        analyze_btn = st.button("🚀 Analyze & Extract", type="primary")

    if analyze_btn:
        text_content = ""
        if uploaded_file:
            with st.spinner("Reading file..."):
                text_content = extract_text_from_file(uploaded_file)
        elif pasted_text:
            text_content = pasted_text
        
        if len(text_content.strip()) > 5:
            start_time = time.time()
            with st.status("Processing NLP...", expanded=True) as status:
                status.write("🔍 Tokenizing & Lemmatizing...")
                data, raw_count = analyze_logic(text_content, min_r, max_r, include_unknown)
                status.write(f"✅ Found {len(data)} unique words.")
                
                st.session_state['gen_words_data'] = data
                st.session_state['raw_count'] = raw_count
                st.session_state['process_time'] = time.time() - start_time
                status.update(label="Analysis Complete", state="complete", expanded=False)
        else:
            st.error("⚠️ Please provide valid text or a file.")

    # Results Display
    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Raw Word Count", f"{st.session_state['raw_count']:,}")
        m2.metric("Target Vocab", f"{len(words_only)}")
        m3.metric("Time Taken", f"{st.session_state['process_time']:.2f}s")
        
        # Preview
        with st.expander("📋 Word List Preview", expanded=False):
            show_rank = st.toggle("Show Rank")
            preview_str = ", ".join([f"{w} ({r})" if show_rank else w for w, r in data_pairs])
            st.markdown(f'<div class="scrollable-text">{preview_str}</div>', unsafe_allow_html=True)
            st.button("📋 Copy List to Clipboard", on_click=lambda: st.write(st.clipboard(preview_str)) or st.toast("Copied!"))

        st.markdown("### ⚙️ Prompt Settings")
        
        # Prompt Config
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            front_mode = st.selectbox("Front Side", ["Word Only", "Phrase/Collocation"])
        with pc2:
            def_mode = st.selectbox("Definition Language", ["English", "中文", "中英双语"])
        with pc3:
            batch_size = st.number_input("Batch Size", 10, 100, 50)
            
        ex_count = st.slider("Example Sentences", 1, 3, 1)
        need_ety = st.checkbox("Include Etymology", value=True)
        
        # Generate Batches
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        st.info(f"Generated {len(batches)} prompt batches.")
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📝 Prompt Batch {idx+1} ({len(batch)} words)"):
                prompt = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.code(prompt, language="text")

# --- Tab 3: Anki Generation ---
with tab_anki:
    st.markdown("### 📦 Generate Anki Package")
    
    st.info("Paste the JSON response from AI here. You can paste multiple responses one after another.")
    
    if 'anki_input_text' not in st.session_state:
        st.session_state['anki_input_text'] = ""
        
    ai_resp = st.text_area("JSON Input", height=200, key="anki_input_text")
    deck_name = st.text_input("Deck Name", f"Vocab_{get_beijing_time_str()}")
    
    if st.button("🛠️ Create .apkg", type="primary"):
        if ai_resp.strip():
            parsed_data = parse_anki_data(ai_resp)
            if parsed_data:
                # Preview Table
                df_view = pd.DataFrame(parsed_data)
                st.write(f"✅ Successfully parsed {len(parsed_data)} cards.")
                st.dataframe(df_view, use_container_width=True, hide_index=True)
                
                # Generate File
                f_path = generate_anki_package(parsed_data, deck_name)
                
                # Download Button
                with open(f_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {deck_name}.apkg",
                        data=f,
                        file_name=f"{deck_name}.apkg",
                        mime="application/octet-stream"
                    )
            else:
                st.error("❌ No valid JSON found. Please check the format.")
        else:
            st.warning("⚠️ Input is empty.")