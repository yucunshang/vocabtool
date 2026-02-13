import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
import zlib  # Added for fixed Deck IDs
from collections import Counter
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# Dynamic Key Initialization
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
    
    /* Scrollable container */
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
    
    /* Guide Styles */
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .guide-title { font-size: 18px; font-weight: bold; color: #0f172a; margin-bottom: 10px; display: block; }
    .guide-tip { font-size: 14px; color: #64748b; background: #eef2ff; padding: 8px; border-radius: 4px; margin-top: 8px; }

    /* Dark Mode Adaptation */
    @media (prefers-color-scheme: dark) {
        .guide-step { background-color: #262730; border-left: 5px solid #4da6ff; box-shadow: none; border: 1px solid #3d3d3d; }
        .guide-title { color: #e0e0e0; }
        .guide-tip { background-color: #31333F; color: #b0b0b0; border: 1px solid #444; }
        .scrollable-text { background-color: #262730; border: 1px solid #444; color: #ccc; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Resource Lazy Loading
# ==========================================
@st.cache_resource(show_spinner="Loading NLP Engine...")
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
    """Load COCA frequency list"""
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
# 2. Extraction Logic (Unchanged)
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
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
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
                    text += soup.get_text(separator=' ', strip=True) + " "
            os.remove(tmp_path)
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
# (UPDATED) Data Parsing Logic for Pipe Format
# ==========================================
def parse_anki_data(raw_text):
    """
    Parses the new Pipe Separated Format (|||)
    Expected: Phrase ||| Meaning ||| Example ||| Etymology
    """
    parsed_cards = []
    
    # Clean up possible Markdown code blocks
    text = raw_text.strip()
    text = re.sub(r'^```.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
    
    lines = text.split('\n')
    seen_phrases = set()

    for line in lines:
        line = line.strip()
        if not line or "|||" not in line: 
            continue
            
        parts = line.split("|||")
        
        # We need at least the phrase and meaning
        if len(parts) < 2: 
            continue
        
        # Extract fields with safe defaults
        w = parts[0].strip() # Natural Phrase
        m = parts[1].strip() # Meaning (English)
        e = parts[2].strip() if len(parts) > 2 else "" # Example
        r = parts[3].strip() if len(parts) > 3 else "" # Etymology (Chinese)

        # Deduplication
        if w.lower() in seen_phrases: 
            continue
        seen_phrases.add(w.lower())
        
        parsed_cards.append({
            'w': w,
            'm': m,
            'e': e,
            'r': r
        })

    return parsed_cards

# ==========================================
# (UPDATED) Anki Generation Logic
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
    
    # CSS Styling
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
    
    # Fixed Model ID to keep card type consistent
    MODEL_ID = 1842957301 
    
    # Fixed Deck ID based on name to preserve progress
    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))

    model = genanki.Model(
        MODEL_ID, 
        'VocabFlow Phrase Model',
        fields=[
            {'name': 'Phrase'},    # w
            {'name': 'Meaning'},   # m
            {'name': 'Example'},   # e
            {'name': 'Etymology'}  # r
        ],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{Phrase}}</div>', 
            'afmt': '''
            {{FrontSide}}
            <hr>
            <div class="meaning">🇬🇧 {{Meaning}}</div>
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
                str(c.get('w', '')), 
                str(c.get('m', '')), 
                str(c.get('e', '')), 
                str(c.get('r', ''))
            ]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# (UPDATED) Prompt Logic - English Only
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    return f"""
Act as a professional lexicographer and Anki card creator. 
I have a list of words. For each word, find a **Natural Phrase** or **Collocation** that best represents its core usage in a native context.

**Input Words:** {w_list}

**Strict Output Format (Pipe Separated):**
1. Do NOT use JSON. Do NOT use Markdown tables.
2. Each line represents one card.
3. Use "|||" as the separator.
4. Structure: `Natural Phrase ||| Concise English Explanation ||| Authentic Example ||| Etymology/Roots (in Chinese)`

**Content Requirements:**
1. **Natural Phrase**: Do not just list the single word. Use a common phrase (e.g., instead of "fan", use "die-hard fan").
2. **Explanation**: Concise, in English.
3. **Example**: Natural, authentic sentence. If multiple examples, join with `<br>`.
4. **Etymology**: In Chinese (Simplified). Explain the root/prefix or a memory aid.

**Example Output:**
well-trained staff ||| having received good training ||| The dog is well-trained and obeys commands. ||| well (好) + train (训练)
look horrified ||| filled with horror; shocked ||| She looked horrified when she saw the bill. ||| 源自拉丁语 horrere (战栗/竖起)

**Start generating now:**
"""

# ==========================================
# 5. UI Main Program
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ Missing `coca_cleaned.csv`")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 Guide", "1️⃣ Extract", "2️⃣ Anki Creation"])

with tab_guide:
    st.markdown("""
    ### 👋 Welcome to Vocab Flow Ultra
    
    **New Workflow:**
    1. **Extract**: Upload your PDF/TXT to extract vocabulary.
    2. **Generate**: Copy the prompt to ChatGPT/Claude/DeepSeek.
    3. **Create**: Paste the **Pipe-separated text (|||)** back here to generate the `.apkg`.
    
    **Note**: JSON is no longer used. The new format is faster and more accurate.
    """)

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 Context Analysis", "🔢 Frequency List"])
    
    with mode_context:
        st.info("💡 **Smart Mode**: Auto lemmatization, deduplication, and cleaning.")
        
        c1, c2 = st.columns(2)
        curr = c1.number_input("Ignore Top N", 1, 20000, 6000, step=100)
        targ = c2.number_input("Ignore Bottom N", 2000, 50000, 10000, step=500)
        include_unknown = st.checkbox("🔓 Include Unknown/Names (Rank > 20000)", value=False)

        uploaded_file = st.file_uploader("📂 Upload (TXT/PDF/DOCX/EPUB)", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("📄 ...or paste text", height=100, key="paste_key")
        
        if st.button("🚀 Start Analysis", type="primary"):
            with st.status("Processing...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 Reading file...")
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 2:
                    status.write("🔍 Analyzing text complexity...")
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, include_unknown)
                    
                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time
                    
                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ Content too short", state="error")
        
        if st.button("🗑️ Clear", type="secondary", on_click=clear_all_state): pass

    with mode_rank:
        gen_type = st.radio("Mode", ["🔢 Sequential", "🔀 Random"], horizontal=True)
        if "Sequential" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("Start Rank", 1, 20000, 1000, step=100)
             count = c_b.number_input("Count", 10, 5000, 50, step=50)
             if st.button("🚀 Generate List"):
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
             min_r = c_min.number_input("Min Rank", 1, 20000, 1, step=100)
             max_r = c_max.number_input("Max Rank", 1, 25000, 5000, step=100)
             r_count = c_cnt.number_input("Count", 10, 5000, 50, step=50)
             if st.button("🎲 Random Draw"):
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

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        st.markdown("### 📊 Report")
        
        k1, k2, k3, k4 = st.columns(4)
        raw_c = st.session_state.get('raw_count', 0)
        stats = st.session_state.get('stats_info', {})
        
        k1.metric("Total Words", f"{raw_c:,}")
        if stats:
            k2.metric("Known Coverage", f"{stats.get('coverage', 0):.1%}")
            k3.metric("Target Density", f"{stats.get('target_density', 0):.1%}")
        else:
            k2.metric("Known Coverage", "--")
            k3.metric("Target Density", "--")
        k4.metric("Extracted", f"{len(words_only)}")
        
        show_rank = st.checkbox("Show Rank", value=False)
        display_text = ", ".join([f"{w}[{r}]" for w, r in data_pairs]) if show_rank else ", ".join(words_only)
            
        with st.expander("📋 **Preview All Words**", expanded=False):
            st.markdown(f'<div class="scrollable-text">{display_text}</div>', unsafe_allow_html=True)
            st.code(display_text, language="text")

        with st.expander("⚙️ **Prompt Settings**", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("Front Mode", ["Natural Phrase", "Word Only"])
            def_mode = col_s2.selectbox("Definition Language", ["English", "Chinese", "Bilingual"])
            ex_count = col_s1.slider("Example Count", 1, 3, 1)
            need_ety = col_s2.checkbox("Include Etymology", value=True)

        batch_size = st.number_input("Batch Size", 50, 500, 150, step=10)
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 Batch {idx+1} ({len(batch)} words)", expanded=(idx==0)):
                prompt_text = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 📦 Create Anki Deck")
    
    if 'anki_cards_cache' not in st.session_state:
        st.session_state['anki_cards_cache'] = None
    
    def reset_anki_state():
        st.session_state['anki_cards_cache'] = None
        if 'anki_input_text' in st.session_state:
             st.session_state['anki_input_text'] = ""

    col_input, col_act = st.columns([3, 1])
    with col_input:
        bj_time_str = get_beijing_time_str()
        deck_name = st.text_input("🏷️ Deck Name", f"Vocab_{bj_time_str}")
    
    st.caption("👇 **Paste the AI output (Pipe separated text |||) here:**")
    
    ai_resp = st.text_area(
        "Input Area", 
        height=300, 
        key="anki_input_text",
        placeholder='well-trained staff ||| having received good training ||| The dog is well-trained... ||| well (好) + train (训练)'
    )

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        start_gen = st.button("🚀 Generate", type="primary", use_container_width=True)
    with c_btn2:
        st.button("🗑️ Reset", type="secondary", on_click=reset_anki_state)

    if start_gen or st.session_state['anki_cards_cache'] is not None:
        if start_gen:
            if not ai_resp.strip():
                st.warning("⚠️ Input is empty.")
            else:
                with st.spinner("Parsing data..."):
                    parsed_data = parse_anki_data(ai_resp)
                    if parsed_data:
                        st.session_state['anki_cards_cache'] = parsed_data
                        st.success(f"✅ Successfully parsed {len(parsed_data)} cards!")
                    else:
                        st.error("❌ Parse failed. Make sure you used the '|||' separator.")
                        st.session_state['anki_cards_cache'] = None

        if st.session_state['anki_cards_cache']:
            cards = st.session_state['anki_cards_cache']
            
            with st.expander("👀 Preview Cards (Top 50)", expanded=True):
                df_view = pd.DataFrame(cards)
                st.dataframe(df_view, use_container_width=True, hide_index=True)

            try:
                f_path = generate_anki_package(cards, deck_name)
                with open(f_path, "rb") as f:
                    file_data = f.read()
                    
                st.download_button(
                    label=f"📥 Download {deck_name}.apkg",
                    data=file_data,
                    file_name=f"{deck_name}.apkg",
                    mime="application/octet-stream",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error generating .apkg: {e}")