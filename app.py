import streamlit as st
import pandas as pd
import re
import os
import io
import time
import lemminflect
import nltk
import genanki
import random
import tempfile
from bs4 import BeautifulSoup

# --- 文件处理库 ---
import pypdf
import docx
import ebooklib
from ebooklib import epub

# ==========================================
# 0. 页面基础配置 & 样式
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
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    .or-divider { text-align: center; margin: 10px 0; color: #888; font-size: 0.9em; font-weight: bold; }
    /* 调整上传组件的内边距 */
    [data-testid='stFileUploader'] { padding-top: 10px; }
    /* 调整按钮间距 */
    .stButton { margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载 & 工具函数
# ==========================================
@st.cache_resource
def setup_nltk():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try: nltk.data.find(f'tokenizers/{pkg}')
            except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except: pass
setup_nltk()

@st.cache_data
def load_vocab_data():
    """加载词频表"""
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

def get_lemma(word):
    try: return lemminflect.getLemma(word, upos='VERB')[0]
    except: return word

def clear_all_state():
    """一键清空的回调函数"""
    st.session_state.clear()

# ==========================================
# 2. 文本提取与分析
# ==========================================
def extract_text_from_file(uploaded_file):
    """多格式文件解析"""
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

def analyze_logic(text, current_lvl, target_lvl):
    """核心分析逻辑"""
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    total_words = len(raw_tokens)
    unique_tokens = set(raw_tokens)
    
    target_words = []
    for w in unique_tokens:
        if len(w) < 3: continue 
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999)
        
        # 筛选: Current < Rank <= Target
        if rank > current_lvl and rank <= target_lvl:
            target_words.append((lemma, rank))
            
    target_words.sort(key=lambda x: x[1])
    final_list = [x[0] for x in target_words]
    return final_list, total_words

# ==========================================
# 3. Anki 生成逻辑
# ==========================================
def generate_anki_package(cards_data, deck_name="Vocab_Deck"):
    # 字体大小: Examples -> 20px, Etymology -> 17px
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2f2f31; color: #f5f5f5; }
    .word { font-size: 40px; font-weight: bold; color: #007AFF; margin-bottom: 10px; }
    .nightMode .word { color: #5FA9FF; }
    .phonetic { color: #888; font-size: 18px; font-family: sans-serif; }
    .def-container { text-align: left; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 10px; }
    .definition { font-weight: bold; color: #444; margin-bottom: 10px; }
    .nightMode .definition { color: #ddd; }
    .examples { background: #f4f4f4; padding: 10px; border-radius: 5px; color: #555; font-style: italic; font-size: 20px; }
    .nightMode .examples { background: #333; color: #ccc; }
    .etymology { margin-top: 15px; font-size: 17px; color: #888; border: 1px dashed #ccc; padding: 5px; display: inline-block;}
    """
    
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31), 'VocabFlow Model',
        fields=[{'name': 'Word'}, {'name': 'IPA'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="word">{{Word}}</div><div class="phonetic">{{IPA}}</div>',
            'afmt': '{{FrontSide}}<div class="def-container"><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div><div class="etymology">🌱 {{Etymology}}</div></div>',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[c['word'], c['ipa'], c['meaning'], c['examples'].replace('\n','<br>'), c['etymology']]))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words):
    """优化后的 Prompt"""
    w_list = ", ".join(words)
    return f"""
You are a strictly compliant dictionary data generator. 
Convert the provided words into Anki card format using the rules below.

**Input Words:** {w_list}

**Strict Output Rules:**
1. **Format:** `Word | IPA | Definition | Examples | Etymology`
2. **Separator:** Use `|` strictly as the field separator. Do NOT use `|` inside the content text.
3. **No Fluff:** Output ONLY the raw text lines. NO headers, NO markdown code blocks, NO conversational filler (e.g., "Here is the list").
4. **Newlines:** Use `<br>` for line breaks inside examples. Do NOT generate actual newlines within a single entry.

**Content Requirements:**
- **IPA:** US pronunciation.
- **Definition:** Simple B2/C1 English. Keep it concise (< 12 words).
- **Examples:** 1 or 2 short, high-context sentences. Separate them with `<br>`. Highlight the keyword in **bold** if possible.
- **Etymology:** Brief root analysis (e.g., "bene(good) + vol(wish)"). If unknown, leave empty.

**Example Output:**
benevolent | /bəˈnevələnt/ | kind and helpful | He was a **benevolent** old man.<br>The fund is for **benevolent** purposes. | bene(good) + vol(wish)
ephemeral | /əˈfemərəl/ | lasting for a very short time | Fashions are **ephemeral**, changing with every season. | epi(on) + hemera(day)
"""

# ==========================================
# 4. 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

# Input Tabs
tab_extract, tab_anki = st.tabs(["1️⃣ 内容提取 & 生成", "2️⃣ 打包 Anki"])

# ------------------------------------------
# TAB 1: 提取逻辑
# ------------------------------------------
with tab_extract:
    # 子 Tab：区分“语境分析”和“纯Rank列表”
    mode_context, mode_rank = st.tabs(["📄 语境分析 (文本/文件)", "🔢 词频列表 (Rank)"])
    
    # --- A. 语境分析模式 ---
    with mode_context:
        st.markdown("#### 1. 设定词汇分级")
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略太简单的 (Current Level)", 1000, 20000, 4000, step=500, help="小于此排名的词会被认为是已掌握词汇")
        targ = c2.number_input("忽略太难的 (Target Level)", 2000, 50000, 15000, step=500, help="只提取此排名内的词")
        
        st.markdown("#### 2. 输入内容 (文件或文本)")
        
        # 统一输入区
        uploaded_file = st.file_uploader("📂 上传文档 (PDF/TXT/DOCX/EPUB)", type=['txt','pdf','docx','epub'])
        
        st.markdown('<div class="or-divider">- OR -</div>', unsafe_allow_html=True)
        
        pasted_text = st.text_area("📄 ...或在此直接粘贴文本", height=150, placeholder="在此处粘贴英文文章...")
        
        # 统一的分析按钮
        if st.button("🚀 开始分析", type="primary"):
            raw_text = ""
            is_file = False
            
            # 优先处理文件
            if uploaded_file:
                with st.spinner(f"正在读取 {uploaded_file.name}..."):
                    raw_text = extract_text_from_file(uploaded_file)
                    is_file = True
            elif pasted_text.strip():
                raw_text = pasted_text
                
            # 执行分析
            if raw_text and len(raw_text) > 10:
                final_words, total = analyze_logic(raw_text, curr, targ)
                st.session_state['gen_words'] = final_words
                st.session_state['total_count'] = total
                if is_file:
                    st.toast(f"文件解析成功，发现 {total} 个词", icon="✅")
            else:
                st.warning("⚠️ 请先上传文件或粘贴文本内容")

        # 移动后的清空按钮：直接显示在分析按钮下方
        if st.button("🗑️ 清空所有数据 (Reset)", type="secondary", on_click=clear_all_state):
            pass

    # --- B. 纯词频生成模式 ---
    with mode_rank:
        st.info("直接从 COCA 词频表中提取指定段落的单词。")
        c_a, c_b = st.columns(2)
        s_rank = c_a.number_input("起始排名 (Start Rank)", 1, 20000, 8000, step=100)
        count = c_b.number_input("生成数量 (Count)", 10, 500, 50, step=10)
        
        if st.button("🔢 生成列表", type="primary"):
            if FULL_DF is not None:
                try:
                    r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                    w_col = next(c for c in FULL_DF.columns if 'word' in c)
                    subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                    st.session_state['gen_words'] = subset[w_col].tolist()
                    st.session_state['total_count'] = count
                except: st.error("数据源格式错误")
        
        # 同样给这里也加一个重置按钮方便操作
        if st.button("🗑️ 清空 (Reset)", type="secondary", key="reset_rank", on_click=clear_all_state):
            pass

    # --- 共通结果展示区 ---
    if 'gen_words' in st.session_state:
        words = st.session_state['gen_words']
        
        st.divider()
        st.markdown(f"""
        <div class="stat-box">
            📊 来源总词数: <b>{st.session_state.get('total_count', 0)}</b> | 
            🎯 筛选后生词: <b>{len(words)}</b> 个
        </div>
        """, unsafe_allow_html=True)

        if len(words) > 0:
            # 结果预览
            with st.expander("👁️ 预览单词列表", expanded=False):
                st.write(", ".join(words))

            st.markdown("### 🤖 获取 AI Prompt")
            c_batch, c_info = st.columns([1, 2])
            
            # 默认 50，上限 200
            batch_size = c_batch.number_input("每组单词数 (Batch Size)", 10, 200, 50, step=10)
            
            c_info.caption(f"💡 建议 30-50 个一组。共需 {len(words)//batch_size + (1 if len(words)%batch_size else 0)} 次对话。")
            
            # 自动分批逻辑
            batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
            
            for idx, batch in enumerate(batches):
                with st.expander(f"第 {idx+1} 组 (单词 {idx*batch_size+1} - {idx*batch_size+len(batch)})", expanded=(idx==0)):
                    prompt = get_ai_prompt(batch)
                    st.code(prompt, language="markdown")
                    st.caption("👆 点击右上角复制 -> 发给 AI -> 复制回复 -> 粘贴到 '打包 Anki' 页面")

# ------------------------------------------
# TAB 2: 打包 Anki
# ------------------------------------------
with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    st.info("💡 提示：将 AI 对话中的回复内容（包含 | 分隔符的行）全部粘贴到下方。支持多次粘贴。")
    
    ai_resp = st.text_area("在此粘贴 AI 的回复内容", height=300, placeholder="word1 | /ipa/ | meaning... \nword2 | ...")
    deck_name = st.text_input("牌组名称 (.apkg)", "VocabFlow_Deck")
    
    if st.button("🔨 生成 .apkg 文件", type="primary"):
        if not ai_resp.strip():
            st.error("❌ 内容为空，请先粘贴 AI 的回复")
        else:
            cards = []
            skipped = 0
            # 宽容解析
            for line in ai_resp.strip().split('\n'):
                line = line.strip()
                if not line: continue
                if "|" not in line: continue
                
                # 增强过滤逻辑：过滤表头和分割线
                if "Word" in line and "IPA" in line: continue  
                if set(line.strip()) == {'-', '|'} or "---" in line: continue 
                
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    cards.append({
                        'word': parts[0],
                        'ipa': parts[1] if len(parts) > 1 else '',
                        'meaning': parts[2] if len(parts) > 2 else '',
                        'examples': parts[3] if len(parts) > 3 else '',
                        'etymology': parts[4] if len(parts) > 4 else ''
                    })
                else:
                    skipped += 1
            
            if cards:
                f_path = generate_anki_package(cards, deck_name)
                with open(f_path, "rb") as f:
                    st.download_button(
                        f"📥 点击下载 {deck_name}.apkg", 
                        f, 
                        file_name=f"{deck_name}.apkg", 
                        mime="application/octet-stream", 
                        type="primary"
                    )
                st.balloons()
                st.success(f"🎉 成功打包 {len(cards)} 张卡片！")
                if skipped > 0:
                    st.caption(f"注：跳过了 {skipped} 行格式不符的数据")
            else:
                st.error("⚠️ 未识别到有效数据，请检查分隔符是否为 '|'")