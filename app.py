import streamlit as st
import pandas as pd
import re
import os
import time
from datetime import datetime, timedelta, timezone
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
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    .or-divider { text-align: center; margin: 10px 0; color: #888; font-size: 0.9em; font-weight: bold; }
    [data-testid='stFileUploader'] { padding-top: 10px; }
    /* 针对 Anki 预览的简单样式 */
    .anki-preview { border: 1px dashed #ccc; padding: 10px; border-radius: 5px; background: #fafafa; margin-bottom: 5px; font-size: 0.9em; }
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
            # 排序并去重，保留排名靠前的
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

def get_beijing_time_str():
    """获取北京时间字符串 (UTC+8)"""
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

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
# 3. Anki 生成逻辑 (修复词源显示)
# ==========================================
def generate_anki_package(cards_data, deck_name):
    # CSS 样式增强：确保词源醒目
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2f2f31; color: #f5f5f5; }
    .word { font-size: 40px; font-weight: bold; color: #007AFF; margin-bottom: 10px; }
    .nightMode .word { color: #5FA9FF; }
    .phonetic { color: #888; font-size: 18px; font-family: sans-serif; margin-bottom: 15px; }
    
    .def-container { text-align: left; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 15px; }
    
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 22px; }
    .nightMode .definition { color: #eee; }
    
    .examples { background: #f4f4f4; padding: 15px; border-radius: 8px; color: #444; font-style: italic; font-size: 20px; line-height: 1.4; margin-bottom: 15px; }
    .nightMode .examples { background: #383838; color: #ddd; }
    
    /* 词源样式增强 */
    .etymology { 
        display: block; 
        font-size: 18px; 
        color: #555; 
        border: 1px dashed #bbb; 
        padding: 8px 12px; 
        border-radius: 6px;
        background-color: #fffaf0;
        margin-top: 10px;
    }
    .nightMode .etymology { 
        color: #aaa; 
        border-color: #555;
        background-color: #333;
    }
    """
    
    # 随机生成 Model ID，防止不同牌组冲突
    model_id = random.randrange(1 << 30, 1 << 31)
    
    model = genanki.Model(
        model_id, 
        f'VocabFlow Model {model_id}',
        fields=[
            {'name': 'Word'}, 
            {'name': 'IPA'}, 
            {'name': 'Meaning'}, 
            {'name': 'Examples'}, 
            {'name': 'Etymology'}
        ],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="word">{{Word}}</div><div class="phonetic">{{IPA}}</div>',
            'afmt': '''
            {{FrontSide}}
            <div class="def-container">
                <div class="definition">{{Meaning}}</div>
                <div class="examples">{{Examples}}</div>
                <div class="etymology">🌱 <b>Etymology:</b> {{Etymology}}</div>
            </div>
            ''',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    
    for c in cards_data:
        deck.add_note(genanki.Note(
            model=model, 
            fields=[
                str(c.get('word','')), 
                str(c.get('ipa','')), 
                str(c.get('meaning','')), 
                str(c.get('examples','')).replace('\n','<br>'), 
                str(c.get('etymology','')) # 确保这里取到了值
            ]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words):
    """Prompt 优化"""
    w_list = ", ".join(words)
    return f"""
Act as a Dictionary API. Convert the following words into strictly formatted data.

**Words:** {w_list}

**CRITICAL FORMATTING RULES (Must Follow):**
1. **Format:** `Word | IPA | Definition | Examples | Etymology`
2. **NO Markdown Tables:** Do NOT use tables. Do NOT use `|` at the start or end of lines.
3. **Separator:** Use `|` ONLY to separate fields.
4. **Content:**
   - Definition: Concise (<12 words).
   - Examples: 2 sentences separated by `<br>`.
   - **Etymology:** REQUIRED. Provide root/suffix analysis (e.g., "bene(good)+vol(wish)"). If unknown, state origin (e.g., "From Old French...").

**Example of CORRECT Output:**
benevolent | /bəˈnevələnt/ | kind and helpful | He is **benevolent**.<br>A **benevolent** fund. | bene(good) + vol(wish)

**Begin Output:**
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
    mode_context, mode_rank = st.tabs(["📄 语境分析 (文本/文件)", "🔢 词频列表 (Rank & Random)"])
    
    # --- A. 语境分析 ---
    with mode_context:
        st.markdown("#### 1. 设定词汇分级")
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略太简单的 (Current Level)", 1000, 20000, 4000, step=500)
        targ = c2.number_input("忽略太难的 (Target Level)", 2000, 50000, 15000, step=500)
        
        st.markdown("#### 2. 输入内容")
        uploaded_file = st.file_uploader("📂 上传文档 (PDF/TXT/DOCX/EPUB)", type=['txt','pdf','docx','epub'])
        st.markdown('<div class="or-divider">- OR -</div>', unsafe_allow_html=True)
        pasted_text = st.text_area("📄 ...或在此直接粘贴文本", height=150)
        
        if st.button("🚀 开始分析", type="primary"):
            raw_text = ""
            if uploaded_file:
                with st.spinner(f"正在读取 {uploaded_file.name}..."):
                    raw_text = extract_text_from_file(uploaded_file)
            elif pasted_text.strip():
                raw_text = pasted_text
                
            if raw_text and len(raw_text) > 10:
                final_words, total = analyze_logic(raw_text, curr, targ)
                st.session_state['gen_words'] = final_words
                st.session_state['total_count'] = total
            else:
                st.warning("⚠️ 请输入有效内容")

        if st.button("🗑️ 清空所有数据 (Reset)", type="secondary", on_click=clear_all_state):
            pass

    # --- B. 纯词频列表 (新增随机功能) ---
    with mode_rank:
        st.info("从 COCA 词频表中生成单词列表。")
        
        # 两个模式：顺序 vs 随机
        gen_type = st.radio("生成模式", ["🔢 顺序截取 (例如: 8000名后的50个)", "🔀 范围随机 (例如: 6000-8000名中随机取50个)"])
        
        if "顺序" in gen_type:
            c_a, c_b = st.columns(2)
            s_rank = c_a.number_input("起始排名 (Start Rank)", 1, 20000, 8000, step=100)
            count = c_b.number_input("数量 (Count)", 10, 500, 50, step=10)
            
            if st.button("🚀 生成顺序列表", type="primary"):
                if FULL_DF is not None:
                    r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                    w_col = next(c for c in FULL_DF.columns if 'word' in c)
                    subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                    st.session_state['gen_words'] = subset[w_col].tolist()
                    st.session_state['total_count'] = count
        else:
            # 随机模式逻辑
            c_min, c_max, c_cnt = st.columns([1,1,1])
            min_r = c_min.number_input("最小排名 (Min)", 1, 20000, 6000, step=500)
            max_r = c_max.number_input("最大排名 (Max)", 1, 25000, 8000, step=500)
            r_count = c_cnt.number_input("随机数量 (Qty)", 10, 200, 50, step=10)
            
            if st.button("🎲 随机抽取", type="primary"):
                if FULL_DF is not None:
                    try:
                        r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                        w_col = next(c for c in FULL_DF.columns if 'word' in c)
                        
                        # 筛选范围
                        mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                        candidates = FULL_DF[mask]
                        
                        avail_count = len(candidates)
                        if avail_count == 0:
                            st.error(f"⚠️ 该范围内 (Rank {min_r}-{max_r}) 没有找到单词。")
                        else:
                            # 随机抽样
                            real_count = min(r_count, avail_count)
                            subset = candidates.sample(n=real_count)
                            # 按Rank排序一下，方便查看
                            subset = subset.sort_values(r_col)
                            
                            st.session_state['gen_words'] = subset[w_col].tolist()
                            st.session_state['total_count'] = real_count
                            st.success(f"成功从 {avail_count} 个候选词中随机抽取了 {real_count} 个！")
                    except Exception as e:
                        st.error(f"生成出错: {e}")

        if st.button("🗑️ 清空 (Reset)", type="secondary", key="reset_rank", on_click=clear_all_state):
            pass

    # --- 结果展示 ---
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
            with st.expander("👁️ 预览单词列表", expanded=False):
                st.write(", ".join(words))

            st.markdown("### 🤖 获取 AI Prompt")
            c_batch, c_info = st.columns([1, 2])
            batch_size = c_batch.number_input("每组单词数", 10, 200, 50, step=10)
            c_info.caption(f"💡 分组建议：每次复制一组给AI，防止生成中断。")
            
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
    
    bj_time_str = get_beijing_time_str()
    default_name = f"Vocab_{bj_time_str}"
    
    if 'anki_input_text' not in st.session_state:
        st.session_state['anki_input_text'] = ""

    ai_resp = st.text_area(
        "在此粘贴 AI 的回复内容 (下载后不会消失，可继续添加)", 
        height=300, 
        placeholder="word1 | /ipa/ | meaning... \nword2 | ...",
        key="anki_input_text"
    )
    
    deck_name = st.text_input("牌组名称 (已自动设为北京时间)", default_name)
    
    # 解析逻辑 (增强容错率，确保词源被捕获)
    cards = []
    skipped = 0
    if ai_resp.strip():
        for line in ai_resp.strip().split('\n'):
            line = line.strip()
            if not line: continue
            
            # 严格过滤无效行
            if line.startswith("|") or line.endswith("|") or "---" in line: continue
            if "Word" in line and "IPA" in line: continue
            
            if "|" not in line: 
                skipped += 1
                continue
            
            # 分割并自动补全缺失的列
            parts = [p.strip() for p in line.split('|')]
            
            # 补全逻辑：如果不够5列，自动补空字符串，防止报错
            while len(parts) < 5:
                parts.append("")
                
            if len(parts) >= 3: # 至少要有 单词|音标|释义
                cards.append({
                    'word': parts[0],
                    'ipa': parts[1],
                    'meaning': parts[2],
                    'examples': parts[3],
                    'etymology': parts[4] # 这里现在一定安全
                })
            else:
                skipped += 1

    # 显示状态与下载
    if cards:
        st.success(f"✅ 已识别 {len(cards)} 张卡片")
        
        # 简单的预览，让用户确认词源是否提取到了
        with st.expander("🔍 检查解析结果 (前3条)"):
            for c in cards[:3]:
                st.markdown(f"**{c['word']}**: {c['etymology'] if c['etymology'] else '❌ 未检测到词源'}")
        
        if skipped > 0:
            st.caption(f"⚠️ 过滤了 {skipped} 行无效数据")
            
        final_filename = f"{deck_name}.apkg"
        f_path = generate_anki_package(cards, deck_name)
        
        with open(f_path, "rb") as f:
            st.download_button(
                label=f"📥 下载 {final_filename}",
                data=f,
                file_name=final_filename,
                mime="application/octet-stream",
                type="primary"
            )
    elif ai_resp.strip():
        st.warning("⚠️ 粘贴内容中未识别到有效卡片，请检查是否包含 '|' 分隔符")