import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
import zlib
import sqlite3
import asyncio
import edge_tts
from collections import Counter
from datetime import datetime, timedelta, timezone

# 尝试导入 OpenAI 用于兼容 DeepSeek
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==========================================
# 0. 页面配置与全局样式
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 初始化 Session State
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"
if 'anki_input_text' not in st.session_state:
    st.session_state['anki_input_text'] = ""
if 'anki_pkg_data' not in st.session_state:
    st.session_state['anki_pkg_data'] = None

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .pro-box { padding: 15px; background-color: #fff9db; border: 1px solid #ffec99; border-radius: 8px; color: #856404; margin: 10px 0; }
    .scrollable-text {
        max-height: 200px; overflow-y: auto; padding: 10px;
        border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;
        font-family: monospace; white-space: pre-wrap;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心资源与文件解析 (保持原样)
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
            except: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except: pass
    return nltk, lemminflect

@st.cache_data
def load_vocab_data():
    for f_name in ["coca_cleaned.csv", "data.csv", "vocab.csv"]:
        if os.path.exists(f_name):
            try:
                df = pd.read_csv(f_name)
                df.columns = [c.strip().lower() for c in df.columns]
                w_col = next(c for c in df.columns if 'word' in c)
                r_col = next(c for c in df.columns if 'rank' in c)
                df = df.dropna(subset=[w_col])
                df[w_col] = df[w_col].astype(str).str.lower().str.strip()
                df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
                df = df.sort_values(r_col).drop_duplicates(subset=[w_col])
                return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
            except: continue
    return {}, None

VOCAB_DICT, FULL_DF = load_vocab_data()

def get_file_parsers():
    import pypdf, docx, ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    import tempfile
    text = ""
    f_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if f_type == 'txt': text = uploaded_file.getvalue().decode('utf-8', 'ignore')
        elif f_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif f_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif f_type == 'epub':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator=' ', strip=True) + " "
            os.remove(tmp_path)
        elif f_type in ['db', 'sqlite']:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
                tmp_db.write(uploaded_file.getvalue()); tmp_db_path = tmp_db.name
            conn = sqlite3.connect(tmp_db_path); cursor = conn.cursor()
            try:
                cursor.execute("SELECT stem FROM WORDS WHERE stem IS NOT NULL")
                text = " ".join([r[0] for r in cursor.fetchall()])
            except: pass
            finally: conn.close(); os.remove(tmp_db_path)
    except Exception as e: return f"Error: {e}"
    return text

# ==========================================
# 2. AI 逻辑 (DeepSeek 内置)
# ==========================================
def call_deepseek_ai(words_list):
    """内置 DeepSeek AI: 极简 Prompt (正面仅单词)"""
    if not OpenAI: return "Error: openai library missing"
    api_key = st.secrets.get("OPENAI_API_KEY")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model_name = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    # 精简版 Prompt
    prompt = f"""Task: Convert words to Anki cards.
Format: Word ||| Definition ||| Sentence
Rule: Field 1 MUST be the isolated word itself. No phrases.

Input:
{", ".join(words_list)}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 3. TTS 与 Anki 打包逻辑 (带避重复功能)
# ==========================================
async def generate_audio_file(text, output_file, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def generate_anki_package(cards, deck_name, enable_tts, voice):
    import genanki, tempfile
    media_files = []
    # 词源字段 F4 设为可选
    model = genanki.Model(1842957301, 'VocabFlow',
        fields=[{'name': 'F1'}, {'name': 'F2'}, {'name': 'F3'}, {'name': 'F4'}, {'name': 'A1'}, {'name': 'A2'}],
        templates=[{'name': 'Card 1', 
                     'qfmt': '<div style="font-size:30px;color:#0056b3;font-weight:bold;">{{F1}}</div><div>{{A1}}</div>',
                     'afmt': '{{FrontSide}}<hr><div style="text-align:left;font-size:18px;"><b>Def:</b> {{F2}}<br><br><i>Ex: {{F3}}</i><br>{{A2}}<br><br>{{#F4}}<small style="color:gray;">🌱 {{F4}}</small>{{/F4}}</div>'}])
    
    deck = genanki.Deck(zlib.adler32(deck_name.encode()), deck_name)
    tmp_dir = tempfile.gettempdir()
    
    prog_bar = st.progress(0, text="正在准备卡片音频...")
    for i, c in enumerate(cards):
        a1, a2 = "", ""
        if enable_tts and c['w']:
            try:
                # 语音1: 正面单词/短语
                f1_name = f"v_{int(time.time())}_{i}_1.mp3"
                f1_path = os.path.join(tmp_dir, f1_name)
                asyncio.run(generate_audio_file(c['w'], f1_path, voice))
                media_files.append(f1_path); a1 = f"[sound:{f1_name}]"
                # 语音2: 例句
                if c['e']:
                    f2_name = f"v_{int(time.time())}_{i}_2.mp3"
                    f2_path = os.path.join(tmp_dir, f2_name)
                    asyncio.run(generate_audio_file(c['e'], f2_path, voice))
                    media_files.append(f2_path); a2 = f"[sound:{f2_name}]"
            except: pass
        deck.add_note(genanki.Note(model=model, fields=[c['w'], c['m'], c['e'], c['r'], a1, a2]))
        prog_bar.progress((i+1)/len(cards), text=f"正在生成音频: {c['w']}")
    
    prog_bar.empty()
    pkg = genanki.Package(deck); pkg.media_files = media_files
    out_apkg = os.path.join(tmp_dir, f"{deck_name}.apkg")
    pkg.write_to_file(out_apkg)
    return out_apkg

def parse_anki_data(raw_text):
    cards = []
    # 提取代码块内容
    content = raw_text
    match = re.search(r'```(?:text|csv)?\s*(.*?)\s*```', raw_text, re.DOTALL)
    if match: content = match.group(1)
    
    for line in content.strip().split('\n'):
        if "|||" in line:
            parts = [p.strip() for p in line.split("|||")]
            if len(parts) >= 2:
                cards.append({
                    'w': parts[0], 
                    'm': parts[1], 
                    'e': parts[2] if len(parts)>2 else "", 
                    'r': parts[3] if len(parts)>3 else ""
                })
    return cards

# ==========================================
# 4. UI 逻辑
# ==========================================
tab_ext, tab_make = st.tabs(["1️⃣ 单词提取", "2️⃣ 卡片制作"])

with tab_ext:
    c1, c2 = st.columns(2)
    min_rank = c1.number_input("忽略高频词 (Min Rank)", 1, 20000, 5000)
    max_rank = c2.number_input("忽略低频词 (Max Rank)", 2000, 50000, 12000)
    
    input_area = st.text_area("✍️ 输入文章、单词列表或上传文件", height=150)
    uploaded = st.file_uploader("支持 Kindle db, PDF, Docx, Epub", key=st.session_state['uploader_id'], label_visibility="collapsed")
    
    if st.button("🚀 分析词汇", type="primary"):
        txt = extract_text_from_file(uploaded) if uploaded else input_area
        if len(txt) > 2:
            nltk, lemminflect = load_nlp_resources()
            raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", txt)
            valid = [t.lower() for t in raw_tokens if len(t)>1 and re.search(r'[aeiouy]', t.lower())]
            counts = Counter(valid)
            final = []
            seen = set()
            for w in valid:
                try: lemma = lemminflect.getLemma(w, upos='VERB')[0]
                except: lemma = w
                rank = VOCAB_DICT.get(lemma, VOCAB_DICT.get(w, 99999))
                if min_rank <= rank <= max_rank and lemma not in seen:
                    final.append((lemma, rank)); seen.add(lemma)
            st.session_state['gen_words_data'] = sorted(final, key=lambda x: x[1])
            st.success(f"✅ 找到 {len(final)} 个生词")
        else: st.warning("内容太少")

    if 'gen_words_data' in st.session_state:
        words_only = [x[0] for x in st.session_state['gen_words_data']]
        st.code(", ".join(words_only))
        
        # --- DeepSeek 内置 AI ---
        st.divider()
        st.subheader("🤖 内置 AI 生成 (DeepSeek)")
        if st.button("✨ 使用 DeepSeek 生成精简卡片", type="primary"):
            with st.status("DeepSeek 正在分批制作卡片...", expanded=True) as status:
                batch_size = 20
                batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
                results = []
                ai_prog = st.progress(0)
                for idx, b in enumerate(batches):
                    ai_prog.progress((idx+1)/len(batches), text=f"正在处理第 {idx+1}/{len(batches)} 组...")
                    results.append(call_deepseek_ai(b))
                st.session_state['anki_input_text'] = "\n".join(results)
                status.update(label="✅ DeepSeek 生成完毕！请前往'卡片制作'页。", state="complete")
                st.balloons()

        # --- Pro 模式推荐 ---
        st.divider()
        st.markdown("""<div class="pro-box">💎 <b>想要最高质量的卡片？</b><br>
        建议使用 <b>GPT-4o / Claude 3.5</b>。请复制下方专业 Prompt，它们能生成带<b>短语搭配</b>和<b>词源解析</b>的专家级卡片。</div>""", unsafe_allow_html=True)
        
        with st.expander("📌 复制专家级 Pro Prompt (推荐使用第三方 AI)"):
            pro_prompt = f"""# Role
You are an expert English Lexicographer and Anki Card Designer. Your goal is to convert a list of target words into high-quality, import-ready Anki flashcards focusing on **natural collocations** (word chunks).
Make sure to process everything in one go, without missing anything.

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
   `Natural Phrase/Collocation` ||| `Concise Definition of the Phrase` ||| `Short Example Sentence` ||| `Etymology breakdown (Simplified Chinese)`

# Field Constraints (Strict)
1. Field 1: Phrase (CRITICAL) - You MUST generate a high-frequency collocation (e.g., If "rain", output "heavy rain").
2. Field 4: Roots/Etymology (Simplified Chinese) - Format: `prefix + root + suffix`.

# Input Data
{", ".join(words_only)}
"""
            st.code(pro_prompt, language="text")

with tab_make:
    deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{datetime.now().strftime('%m%d_%H%M')}")
    # 绑定到 session_state 以便 AI 自动填充
    ai_output = st.text_area("📥 粘贴 AI 内容 (内置 AI 生成后会自动填入)", value=st.session_state['anki_input_text'], height=250)
    
    col_v1, col_v2 = st.columns(2)
    # 默认开启语音
    use_tts = col_v1.checkbox("🔊 开启语音合成", value=True)
    voice_sex = col_v2.selectbox("🎙️ 选择发音", ["👩 女声 (Jenny)", "👨 男声 (Christopher)"])
    v_code = "en-US-JennyNeural" if "女声" in voice_sex else "en-US-ChristopherNeural"

    if st.button("📦 制作并下载 Anki 牌组", type="primary"):
        cards_list = parse_anki_data(ai_output)
        if cards_list:
            with st.spinner("正在打包中，请稍候..."):
                apkg_path = generate_anki_package(cards_list, deck_name, use_tts, v_code)
                with open(apkg_path, "rb") as f:
                    st.session_state['anki_pkg_data'] = f.read()
                    st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"
                st.success("打包成功！")
        else: st.error("无法解析内容，请检查分隔符是否为 |||")

    # 只有当数据存在时才显示下载按钮，防止点击下载时重新运行生成逻辑
    if st.session_state['anki_pkg_data']:
        st.download_button(
            label=f"💾 点击下载 {st.session_state['anki_pkg_name']}",
            data=st.session_state['anki_pkg_data'],
            file_name=st.session_state['anki_pkg_name'],
            mime="application/octet-stream",
            type="primary"
        )