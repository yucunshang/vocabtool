import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import genanki
import random
import tempfile

# ==========================================
# 0. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Pro (English Def)", 
    page_icon="⚡️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 注入 CSS：美化界面
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; }
    .success-box { padding: 10px; background-color: #e6fffa; border-radius: 5px; color: #006d5b; margin-bottom: 10px; }
    .info-text { font-size: 0.9em; color: #555; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载 (适配 Streamlit Cloud)
# ==========================================
@st.cache_resource
def setup_nltk():
    """在云端环境下安全下载 NLTK 数据"""
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        st.warning(f"NLTK 初始化警告: {e}")

setup_nltk()

@st.cache_data
def load_vocab_data():
    """加载词频数据，增加容错"""
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # 简单的列名清洗
            df.columns = [c.strip().lower() for c in df.columns]
            # 自动寻找 word 和 rank 列
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            
            # 去重保留 rank 最小的
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except Exception as e:
            st.error(f"词频文件读取失败: {e}")
            return {}
    return {}

VOCAB_DICT = load_vocab_data()

def get_lemma(word):
    try: return lemminflect.getLemma(word, upos='VERB')[0]
    except: return word

# ==========================================
# 2. 核心分析逻辑 (保留 Current/Target 筛选)
# ==========================================
def analyze_text(text, current_lvl, target_lvl):
    raw_words = re.findall(r"[a-z]+", text.lower())
    unique_words = set(raw_words)
    
    target_words = [] 
    mastered_count = 0
    beyond_count = 0
    
    for w in unique_words:
        if len(w) < 2: continue
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999) 
        
        # --- 筛选逻辑 ---
        if rank <= current_lvl:
            mastered_count += 1
        elif rank <= target_lvl:
            target_words.append((lemma, rank))
        else:
            beyond_count += 1
            
    target_words.sort(key=lambda x: x[1])
    final_list = [x[0] for x in target_words]
    
    return final_list, mastered_count, beyond_count

# ==========================================
# 3. Anki 打包逻辑 (生成 .apkg)
# ==========================================
def generate_anki_package(cards_data, deck_name="Vocab_Deck"):
    """
    生成高质量 Anki 包，内置 CSS 适配 iOS 深色模式
    """
    
    # --- 高质量 CSS 模板 (英英风格) ---
    CSS = """
    .card {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        text-align: center;
        font-size: 20px;
        color: #333;
        background-color: #ffffff;
        padding: 20px 10px;
    }
    
    .nightMode .card { background-color: #2f2f31; color: #f5f5f5; }
    
    /* 正面 */
    .word { font-size: 38px; font-weight: 700; color: #007AFF; margin-bottom: 8px; }
    .nightMode .word { color: #5FA9FF; }
    .phonetic { font-family: "Lucida Sans Unicode", sans-serif; color: #888; font-size: 18px; }
    
    /* 背面 */
    .def-container { 
        text-align: left; margin-top: 20px; padding-top: 15px; 
        border-top: 1px solid #eee; 
    }
    .nightMode .def-container { border-top: 1px solid #444; }
    
    /* 英文释义样式 */
    .definition { font-weight: 500; font-size: 18px; color: #333; margin-bottom: 15px; line-height: 1.5; }
    .nightMode .definition { color: #eee; }
    
    .example-box {
        background: #f2f7fa; border-left: 4px solid #007AFF;
        padding: 10px; margin: 10px 0; border-radius: 4px;
        font-size: 16px; color: #555; text-align: left; font-style: italic;
    }
    .nightMode .example-box { background: #333333; border-left: 4px solid #5FA9FF; color: #ccc; }
    
    .etymology {
        margin-top: 20px; font-size: 15px; color: #666; 
        border: 1px dashed #bbb; padding: 8px; border-radius: 6px; display: block;
        text-align: left;
    }
    .nightMode .etymology { border-color: #555; color: #aaa; }
    .root-highlight { font-weight: bold; color: #d63031; }
    .nightMode .root-highlight { color: #ff7675; }
    """

    # --- Anki Model 定义 ---
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'VocabFlow English Model',
        fields=[
            {'name': 'Word'},
            {'name': 'IPA'},
            {'name': 'Meaning'},
            {'name': 'Examples'},
            {'name': 'Etymology'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '<div class="word">{{Word}}</div><div class="phonetic">{{IPA}}</div>',
                'afmt': '''
                {{FrontSide}}
                <div class="def-container">
                    <div class="definition">{{Meaning}}</div>
                    <div class="example-box">{{Examples}}</div>
                    <div class="etymology">🌱 <b>Roots & Affixes:</b><br>{{Etymology}}</div>
                </div>
                ''',
            },
        ],
        css=CSS
    )

    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)

    for card in cards_data:
        deck.add_note(genanki.Note(
            model=model,
            fields=[
                card['word'],
                card['ipa'],
                card['meaning'],
                card['examples'].replace('\n', '<br>'),
                card['etymology']
            ]
        ))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt 生成器 (English Version)
# ==========================================
def get_ai_prompt(words):
    w_list = ", ".join(words)
    return f"""
Act as an expert Etymologist and Lexicographer. Create Anki card data.
Words: {w_list}

**Strict Output Format (Pipe Separated `|`, NO Header):**
Word | IPA | Concise English Definition | 2 English Sentences | Etymology (Roots/Affixes)

**Requirements:**
1. **Definition**: Concise English definition (B2/C1 level). Keep it short.
2. **Examples**: 2 authentic English sentences. Use `<br>` to separate them.
3. **Etymology**: Break down the word into roots/affixes. Explain the meaning of the root. 
   - Format: `root(meaning) + suffix(function)`
   - Example: `bene(good) + vol(wish)`
4. **No Header Row**.

**Example Line:**
benevolent | /bəˈnevələnt/ | well meaning and kindly | He was a benevolent old man.<br>The fund provided benevolent assistance. | bene(good) + vol(wish) + -ent(adj suffix)
"""

# ==========================================
# 5. 主界面逻辑
# ==========================================
st.title("⚡️ Vocab Flow (Eng Def)")
st.caption("文本分析 -> 筛选 -> AI 生成 (英英释义+词源) -> iOS Anki 包")

if not VOCAB_DICT:
    st.error("⚠️ 未在目录下检测到 `coca_cleaned.csv`，无法进行词频筛选！")

t1, t2 = st.tabs(["1️⃣ 分析与提词", "2️⃣ 生成 Anki 包"])

# --- Tab 1: 分析 ---
with t1:
    c1, c2 = st.columns(2)
    curr_lvl = c1.number_input("当前词汇量 (Current)", 1000, 20000, 4000, step=500, help="忽略太简单的词")
    targ_lvl = c2.number_input("目标词汇量 (Target)", 1000, 30000, 12000, step=500, help="忽略太生僻的词")
    
    txt = st.text_area("在此粘贴英文文本/文章", height=150, placeholder="Paste English text here...")
    
    if st.button("🔍 开始分析", type="primary"):
        if not txt.strip():
            st.warning("请先输入文本")
        elif not VOCAB_DICT:
            st.warning("无词频数据，无法筛选")
        else:
            final_words, num_m, num_b = analyze_text(txt, curr_lvl, targ_lvl)
            st.session_state['words'] = final_words
            
            st.markdown(f"""
            <div class="success-box">
                <b>🎯 筛选出 {len(final_words)} 个重点词 (Learning Zone)</b><br>
                <span style='font-size:0.85em; opacity:0.8'>
                ✅ 已掌握: {num_m} | 🚀 超纲: {num_b}
                </span>
            </div>
            """, unsafe_allow_html=True)

    if 'words' in st.session_state and st.session_state['words']:
        words_str = st.text_area("确认单词列表", ", ".join(st.session_state['words']), height=100)
        
        st.markdown("##### 🚀 复制 Prompt 发给 AI")
        if st.button("生成 English Prompt"):
            final_list = [w.strip() for w in words_str.split(',') if w.strip()]
            prompt = get_ai_prompt(final_list)
            st.code(prompt, language="markdown")
            st.info("💡 将 AI 回复的管道符内容复制，去 Tab 2 生成卡片。")

# --- Tab 2: 制作 ---
with t2:
    st.markdown("### 🛠️ 制作 iOS 完美适配包")
    st.markdown("<div class='info-text'>将 AI 回复粘贴到下方 (格式: Word | IPA | Def | Ex | Etym)：</div>", unsafe_allow_html=True)
    
    ai_response = st.text_area("粘贴 AI 数据", height=200, placeholder="benevolent | ... | well meaning | ... | bene(good)+vol(wish)")
    deck_title = st.text_input("牌组名称", "My English Vocab")
    
    if st.button("📦 生成 .apkg 文件", type="primary"):
        if not ai_response.strip():
            st.error("内容为空")
        else:
            lines = ai_response.strip().split('\n')
            cards = []
            err_cnt = 0
            
            for line in lines:
                if "|" not in line: continue
                if "Word | IPA" in line: continue 
                
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
                    err_cnt += 1
            
            if cards:
                try:
                    tmp_file_path = generate_anki_package(cards, deck_title)
                    with open(tmp_file_path, "rb") as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label=f"📥 点击下载 {deck_title}.apkg",
                        data=file_data,
                        file_name=f"{deck_title}.apkg",
                        mime="application/octet-stream",
                        type="primary"
                    )
                    st.success(f"成功打包 {len(cards)} 张卡片！")
                except Exception as e:
                    st.error(f"打包出错: {e}")
            else:
                st.error("数据格式错误，请检查分隔符 |")