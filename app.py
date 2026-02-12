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
    .stat-box { text-align: center; padding: 10px; background: #f1f5f9; border-radius: 8px; }
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

@st.cache_resource(show_spinner="正在初始化 NLP 智能引擎...")
def load_nlp_resources():
    """
    按需加载 NLP 库 & 词性标注模型
    """
    import nltk
    import lemminflect
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # 核心包：Tokenizer, Tagger(词性), Wordnet(还原)
    required_packages = ['averaged_perceptron_tagger', 'punkt', 'punkt_tab', 'wordnet']
    
    for pkg in required_packages:
        try: 
            # 尝试查找不同路径下的资源
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError: 
            try: nltk.data.find(f'taggers/{pkg}')
            except LookupError: 
                try: nltk.data.find(f'corpora/{pkg}')
                except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
            
    return nltk, lemminflect

@st.cache_data
def load_vocab_db():
    """
    加载 COCA 词频表
    """
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
        
        # 去重，保留排名最靠前的 (e.g. 动词 wind 和名词 wind，保留高频的)
        df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
    except Exception as e:
        st.error(f"词库加载失败: {e}")
        return {}, None

VOCAB_DICT, FULL_DF = load_vocab_db()

def read_file_content(uploaded_file):
    """
    鲁棒的文件读取器 (PDF/Docx/Epub/Txt)
    """
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
    V29 Plus 算法：NLP POS 过滤 + 智能还原 + 词频比对
    """
    nltk, lemminflect = load_nlp_resources()
    
    # 1. 预清洗：去除 URL、邮箱、数字乱码
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    
    # 2. NLTK 智能分词
    raw_tokens = nltk.word_tokenize(text)
    total_words = len(raw_tokens)
    
    # 3. 词性标注 (POS Tagging) - 核心精度来源
    tagged_tokens = nltk.pos_tag(raw_tokens)
    
    # 允许的词性: 名词(N), 动词(V), 形容词(J), 副词(R)
    # 拒绝: 代词, 介词, 连词, 冠词等
    VALID_PREFIXES = ('N', 'V', 'J', 'R') 
    
    candidates = []
    seen_lemmas = set()
    
    for word, tag in tagged_tokens:
        # A. 基础过滤：全字母，长度>1
        if len(word) < 2 or not word.isalpha():
            continue
            
        # B. 词性过滤
        if not tag.startswith(VALID_PREFIXES):
            continue
            
        # C. 智能还原 (Lemma)
        # 将 Treebank Tag 映射到 Lemminflect UPOS
        if tag.startswith('V'): upos = 'VERB'
        elif tag.startswith('J'): upos = 'ADJ'
        elif tag.startswith('R'): upos = 'ADV'
        else: upos = 'NOUN'
        
        lemma = lemminflect.getLemma(word.lower(), upos=upos)[0]
        
        # D. 查词频 (优先查 Lemma)
        rank_l = VOCAB_DICT.get(lemma, 99999)
        rank_w = VOCAB_DICT.get(word.lower(), 99999)
        
        # 确定最佳 Rank (如果两者都有排名，取更靠前的)
        if rank_l != 99999 and rank_w != 99999:
            best_rank = min(rank_l, rank_w)
        elif rank_l != 99999:
            best_rank = rank_l
        else:
            best_rank = rank_w
            
        # E. 范围判定
        in_range = cfg['curr'] <= best_rank <= cfg['targ']
        is_unknown = (best_rank == 99999 and cfg['include_unknown'])
        
        if in_range or is_unknown:
            # 过滤全大写缩写 (如 API, HTML)，除非它是已知高频词
            if word.isupper() and best_rank > 5000:
                continue

            # 最终展示词 (优先展示 Lemma)
            display_word = lemma if rank_l != 99999 else word.lower()
            
            if display_word not in seen_lemmas:
                candidates.append((display_word, best_rank))
                seen_lemmas.add(display_word)

    # 排序：按词频 (常见 -> 生僻 -> 未知)
    return sorted(candidates, key=lambda x: x[1]), total_words

def create_anki_pkg(cards, deck_name):
    """
    生成 Anki .apkg 文件
    """
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

def get_ai_prompt(words, settings):
    """
    生成 AI 提示词 (NDJSON 格式)
    """
    w_str = ", ".join(words)
    context_desc = "short phrase/collocation (2-5 words)" if settings['front'] == "短语" else "word itself"
    
    return f"""
Task: Create Anki cards for learning English.
Target Words: {w_str}

**OUTPUT FORMAT: NDJSON (One JSON object per line).**
**Strictly NO markdown code blocks (```json ... ```). Just raw NDJSON.**

**Fields:**
1. `w` (Front): The {context_desc} containing the target word.
2. `m` (Meaning): {settings['def_lang']} definition.
3. `e` (Examples): {settings['ex_count']} example sentence(s).
4. `r` (Etymology): {"Root/Etymology (Simple Chinese)" if settings['ety'] else "Empty string"}.

**Example:**
{{"w": "serendipity", "m": "意外发现珍宝的运气", "e": "It was pure serendipity.", "r": "from Horace Walpole"}}

**Start:**
"""

# ==========================================
# 2. Main App Logic
# ==========================================

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.warning("⚠️ 未检测到 `coca_cleaned.csv`，仅可使用无词频过滤模式。")

# 初始化 Session State
if 'uploader_id' not in st.session_state: st.session_state['uploader_id'] = "1000"
if 'gen_data' not in st.session_state: st.session_state['gen_data'] = []
if 'raw_count' not in st.session_state: st.session_state['raw_count'] = 0

tab_guide, tab_extract, tab_anki = st.tabs(["📖 指南", "1️⃣ 提取", "2️⃣ Anki"])

with tab_guide:
    st.markdown("""
    <div class="guide-step">
        <span class="guide-title">Step 1: 提取生词 (Extract)</span>
        上传 PDF/EPUB/TXT。系统利用 NLP 算法自动剔除 <code>the, of, is</code> 等虚词，并智能还原时态。
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

with tab_extract:
    st.info("💡 **Vibe Update**: 已启用 NLP 词性分析。系统将自动忽略代词、介词、连词，仅保留核心实词。")
    
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
            with st.status("正在进行 NLP 分析...", expanded=True) as status:
                status.write("🧠 正在加载模型并进行分词...")
                data, total = process_text_logic(raw_text, {'curr': curr, 'targ': targ, 'include_unknown': include_unknown})
                st.session_state['gen_data'] = data
                st.session_state['raw_count'] = total
                status.update(label=f"✅ 分析完成！从 {total} 词中提取出 {len(data)} 个生词", state="complete", expanded=False)

    # 结果展示区
    if st.session_state['gen_data']:
        data = st.session_state['gen_data']
        words = [x[0] for x in data]
        
        st.divider()
        c_k1, c_k2 = st.columns(2)
        c_k1.metric("文档总词数", st.session_state.get('raw_count', 0))
        c_k2.metric("生词提取数", len(data))

        with st.expander("⚙️ Prompt 设置 (点击展开)", expanded=True):
            cols = st.columns(4)
            s_front = cols[0].selectbox("正面内容", ["单词", "短语"], index=0)
            s_def = cols[1].selectbox("释义语言", ["中文", "英文", "中英"], index=0)
            s_ex = cols[2].slider("例句数量", 1, 3, 1)
            s_ety = cols[3].checkbox("包含词源", True)
            
            settings = {'front': s_front, 'def_lang': s_def, 'ex_count': s_ex, 'ety': s_ety}

        # 预览与复制区
        st.markdown("### 📋 生成 Prompt")
        batch_size = st.number_input("每组单词数量 (Batch Size)", 10, 500, 50)
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
        
        for i, batch in enumerate(batches):
            prompt = get_ai_prompt(batch, settings)
            st.text_area(f"第 {i+1} 组 (共 {len(batch)} 词)", value=prompt, height=100, key=f"p_{i}")

        if st.button("🗑️ 清空所有数据", type="secondary"):
            st.session_state.clear()
            st.rerun()

with tab_anki:
    st.markdown("### 📦 JSON 转 Anki")
    json_input = st.text_area("在此粘贴 AI 回复的 JSON (支持多次追加)", height=200)
    deck_name = st.text_input("牌组名称", f"Vocab_{datetime.now().strftime('%m%d')}")

    if json_input:
        try:
            # 宽松解析: 提取所有 {} 包裹的内容
            matches = re.findall(r'\{.*?\}', json_input, re.DOTALL)
            parsed = []
            for m in matches:
                try: parsed.append(json.loads(m))
                except: pass
            
            clean_cards = []
            for p in parsed:
                if 'w' in p and 'm' in p:
                    clean_cards.append({
                        'w': p.get('w'), 'm': p.get('m'), 
                        'e': p.get('e', ''), 'r': p.get('r', '')
                    })
            
            if clean_cards:
                st.success(f"✅ 解析成功: {len(clean_cards)} 张卡片")
                # 预览前5条
                st.dataframe(pd.DataFrame(clean_cards).head(5)[['w', 'm', 'r']], hide_index=True, use_container_width=True)
                
                pkg_path = create_anki_pkg(clean_cards, deck_name)
                with open(pkg_path, "rb") as f:
                    st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", type="primary")
            else:
                st.warning("⚠️ 未找到有效的 JSON 对象，请检查粘贴内容。")
        except Exception as e:
            st.error(f"解析错误: {e}")