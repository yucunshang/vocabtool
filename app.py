import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. Vibe Config & Constants
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .scrollable-text {
        max-height: 200px; overflow-y: auto; padding: 10px;
        border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;
        font-family: monospace; white-space: pre-wrap;
    }
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; }
    .guide-title { font-size: 18px; font-weight: bold; color: #0f172a; display: block; margin-bottom: 8px;}
</style>
"""

ANKI_CSS = """
.card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
.nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
.phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
.definition { font-weight: bold; color: #222; margin-bottom: 15px; text-align: left; }
.examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; font-style: italic; text-align: left; margin-bottom: 15px;}
.etymology { font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border: 1px solid #fef3c7; border-radius: 6px; }
"""

# ==========================================
# 1. Core Services (Logic Only)
# ==========================================

@st.cache_resource(show_spinner="正在加载 NLP 引擎...")
def load_nlp():
    """按需加载重型 NLP 库"""
    import nltk
    import lemminflect
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
        try: nltk.data.find(f'tokenizers/{pkg}')
        except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
            
    return lemminflect

@st.cache_data
def load_vocab_db():
    """加载 COCA 词频表"""
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    if not file_path: return {}, None

    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower() for c in df.columns]
        w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
        r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
        
        df = df.dropna(subset=[w_col])
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
        
        # 去重，保留排名最高的
        df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
    except Exception as e:
        st.error(f"词库加载失败: {e}")
        return {}, None

VOCAB_DICT, FULL_DF = load_vocab_db()

def read_file_content(uploaded_file):
    """鲁棒的文件读取器"""
    import pypdf, docx, ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    ftype = uploaded_file.name.split('.')[-1].lower()
    try:
        if ftype == 'txt':
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ftype == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ftype == 'docx':
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ftype == 'epub':
            # EPUB 处理略繁琐，需临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            text = ""
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator=' ', strip=True) + " "
            os.remove(tmp_path)
            return text
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return ""

def process_text_logic(text, cfg):
    """
    核心业务逻辑：文本 -> 清洗 -> 还原 -> 过滤 -> 排序
    cfg: {curr, targ, include_unknown}
    """
    lemminflect = load_nlp()
    
    # 1. 粗分词
    tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(tokens)
    
    # 2. 预处理 (过滤短词、垃圾词)
    valid_tokens = {t.lower() for t in tokens if len(t) >= 2 and not re.search(r'(.)\1{2,}', t)}
    
    candidates = []
    seen_lemmas = set()
    
    for w in valid_tokens:
        # 获取 Lemma 和 Rank
        lemma = lemminflect.getLemma(w, upos='VERB')[0]
        rank_l = VOCAB_DICT.get(lemma, 99999)
        rank_w = VOCAB_DICT.get(w, 99999)
        best_rank = min(rank_l, rank_w) if rank_l != 99999 and rank_w != 99999 else (rank_l if rank_l != 99999 else rank_w)

        # 过滤逻辑
        in_range = cfg['curr'] <= best_rank <= cfg['targ']
        is_unknown = (best_rank == 99999 and cfg['include_unknown'])
        
        if in_range or is_unknown:
            # 优先展示 Lemma
            display_word = lemma if rank_l != 99999 else w
            if display_word not in seen_lemmas:
                candidates.append((display_word, best_rank))
                seen_lemmas.add(display_word)
                
    return sorted(candidates, key=lambda x: x[1]), total_words

def create_anki_pkg(cards, deck_name):
    import genanki, tempfile
    
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'VocabFlow Model',
        fields=[{'name': 'Front'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="phrase">{{Front}}</div>',
            'afmt': '{{FrontSide}}<hr><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div>{{#Etymology}}<div class="etymology">🌱 {{Etymology}}</div>{{/Etymology}}',
        }],
        css=ANKI_CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards:
        deck.add_note(genanki.Note(model=model, fields=[c['w'], c['m'], c['e'].replace('\n','<br>'), c['r']]))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 2. UI Components
# ==========================================

def render_guide():
    st.markdown("""
    <div class="guide-step">
        <span class="guide-title">Step 1: 提取生词 (Extract)</span>
        上传 PDF/EPUB/TXT，设置词频过滤范围（如忽略前 2000 个高频词）。
    </div>
    <div class="guide-step">
        <span class="guide-title">Step 2: 获取 Prompt (AI Generation)</span>
        复制生成的 Prompt 发送给 AI (ChatGPT/Claude)，获取 JSON 数据。
    </div>
    <div class="guide-step">
        <span class="guide-title">Step 3: 制作 Anki (Create Deck)</span>
        将 AI 返回的 JSON 粘贴回来，一键打包下载 .apkg 文件。
    </div>
    """, unsafe_allow_html=True)

def get_ai_prompt(words, settings):
    """构建 Prompt"""
    w_str = ", ".join(words)
    context_desc = "phrase/collocation" if settings['front'] == "短语" else "word itself"
    
    return f"""
Task: Create Anki cards for learning English.
Words: {w_str}

**OUTPUT FORMAT: NDJSON (One JSON object per line, no markdown wrapper).**

**Fields:**
1. `w` (Front): The {context_desc}.
2. `m` (Meaning): {settings['def_lang']} definition.
3. `e` (Examples): {settings['ex_count']} example sentence(s).
4. `r` (Etymology): {"Root/Etymology (Simple Chinese)" if settings['ety'] else "Empty string"}.

**Example Line:**
{{"w": "serendipity", "m": "意外发现珍宝的运气", "e": "It was pure serendipity that we met.", "r": "from Horace Walpole"}}

**Start:**
"""

# ==========================================
# 3. Main App Flow
# ==========================================

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.warning("⚠️ 未检测到 `coca_cleaned.csv`，仅可使用无词频过滤模式。")

# State Init
if 'uploader_id' not in st.session_state: st.session_state['uploader_id'] = "1000"
if 'gen_data' not in st.session_state: st.session_state['gen_data'] = []

tab_guide, tab_extract, tab_anki = st.tabs(["📖 指南", "1️⃣ 提取", "2️⃣ Anki"])

with tab_guide:
    render_guide()

with tab_extract:
    c1, c2 = st.columns(2)
    curr = c1.number_input("忽略前 N 高频词", 0, 20000, 2000, step=100)
    targ = c2.number_input("忽略后 N 生僻词", 2000, 60000, 20000, step=500)
    include_unknown = st.checkbox("🔓 包含未收录词 (人名/新词)", False)

    uploaded = st.file_uploader("📄 拖入文档 (PDF/EPUB/TXT)", key=st.session_state['uploader_id'])
    text_input = st.text_area("...或直接粘贴文本", height=100)

    if st.button("🚀 开始分析", type="primary"):
        raw_text = read_file_content(uploaded) if uploaded else text_input
        if len(raw_text) < 5:
            st.error("⚠️ 内容太短，请重新输入")
        else:
            with st.status("正在处理...", expanded=True) as status:
                status.write("🔍 清洗与NLP还原...")
                data, total = process_text_logic(raw_text, {'curr': curr, 'targ': targ, 'include_unknown': include_unknown})
                st.session_state['gen_data'] = data
                st.session_state['raw_count'] = total
                status.update(label=f"✅ 完成！提取到 {len(data)} 个生词", state="complete", expanded=False)

    # 结果展示区
    if st.session_state['gen_data']:
        data = st.session_state['gen_data']
        words = [x[0] for x in data]
        
        st.divider()
        c_k1, c_k2 = st.columns(2)
        c_k1.metric("文档总词数", st.session_state.get('raw_count', 0))
        c_k2.metric("生词提取数", len(data))

        with st.expander("⚙️ Prompt 设置", expanded=True):
            cols = st.columns(4)
            s_front = cols[0].selectbox("正面", ["单词", "短语"], index=0)
            s_def = cols[1].selectbox("释义", ["中文", "英文", "中英"], index=0)
            s_ex = cols[2].slider("例句数", 1, 3, 1)
            s_ety = cols[3].checkbox("词源", True)
            
            settings = {'front': s_front, 'def_lang': s_def, 'ex_count': s_ex, 'ety': s_ety}

        # 预览与复制
        batch_size = st.number_input("每组单词数", 10, 500, 50)
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
        
        st.caption(f"共分为 {len(batches)} 组 Prompt。请依次复制发给 AI。")
        
        for i, batch in enumerate(batches):
            prompt = get_ai_prompt(batch, settings)
            st.text_area(f"Batch {i+1}", value=prompt, height=80, key=f"p_{i}")

        if st.button("🗑️ 清空所有", type="secondary"):
            st.session_state.clear()
            st.rerun()

with tab_anki:
    st.markdown("### 📦 JSON 转 Anki")
    json_input = st.text_area("在此粘贴 AI 回复的 JSON (支持多次追加)", height=200)
    deck_name = st.text_input("牌组名称", f"Vocab_{datetime.now().strftime('%m%d')}")

    if json_input:
        try:
            # 宽松的 JSON 提取正则
            matches = re.findall(r'\{.*?\}', json_input, re.DOTALL)
            parsed = [json.loads(m) for m in matches]
            
            clean_cards = []
            for p in parsed:
                if 'w' in p and 'm' in p:
                    clean_cards.append({
                        'w': p.get('w'), 'm': p.get('m'), 
                        'e': p.get('e', ''), 'r': p.get('r', '')
                    })
            
            if clean_cards:
                st.success(f"解析成功: {len(clean_cards)} 张卡片")
                st.dataframe(pd.DataFrame(clean_cards).head(5))
                
                pkg_path = create_anki_pkg(clean_cards, deck_name)
                with open(pkg_path, "rb") as f:
                    st.download_button("📥 下载 .apkg", f, file_name=f"{deck_name}.apkg", type="primary")
            else:
                st.warning("未找到有效的 JSON 对象")
        except Exception as e:
            st.error(f"解析错误: {e}")