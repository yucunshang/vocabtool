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
# 0. 页面配置 & 样式优化
# ==========================================
st.set_page_config(
    page_title="Vocab Flow (Server Ver.)", 
    page_icon="⚡️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 注入 CSS 美化 Streamlit 界面
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .instruction { font-size: 0.9em; color: #666; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载 (适配云端环境)
# ==========================================
@st.cache_resource
def setup_nltk():
    """在云端环境下安全下载 NLTK 数据"""
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # 仅下载必要的包
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        st.warning(f"NLTK Setup Warning: {e}")

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
            # 尝试自动寻找 word 和 rank 列
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            
            # 去重保留 rank 最小的
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except:
            return {}
    return {}

VOCAB_DICT = load_vocab_data()

def get_lemma(word):
    """获取单词原形"""
    try:
        return lemminflect.getLemma(word, upos='VERB')[0]
    except:
        return word

# ==========================================
# 2. Anki 高质量模板与打包逻辑
# ==========================================
def generate_anki_package(cards_data, deck_name="Vocab_Deck"):
    """
    生成 .apkg 文件并返回二进制数据
    cards_data: list of dicts
    """
    
    # --- CSS 样式 (高质量模板核心) ---
    # 这个样式会自动适配 iOS 的夜间模式
    CSS = """
    .card {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        text-align: center;
        font-size: 20px;
        color: #333;
        background-color: #ffffff;
        padding: 20px 10px;
    }
    
    /* 夜间模式适配 */
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
    
    .definition { font-weight: 600; font-size: 18px; color: #444; margin-bottom: 15px; }
    .nightMode .definition { color: #ddd; }
    
    .example-box {
        background: #f2f7fa; border-left: 4px solid #007AFF;
        padding: 10px; margin: 10px 0; border-radius: 4px;
        font-size: 16px; color: #555; text-align: left;
    }
    .nightMode .example-box { background: #333333; border-left: 4px solid #5FA9FF; color: #ccc; }
    
    .etymology {
        margin-top: 20px; font-size: 14px; color: #999; font-style: italic;
        border: 1px dashed #ddd; padding: 5px; border-radius: 5px; display: inline-block;
    }
    .nightMode .etymology { border-color: #555; }
    """

    # --- Anki Model 定义 ---
    # 字段：Word, IPA, Meaning, Examples, Etymology
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'Streamlit High-End Model',
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
                    <div class="etymology">Origin: {{Etymology}}</div>
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
                card['examples'].replace('\n', '<br>'), # 处理换行
                card['etymology']
            ]
        ))

    # 使用临时文件生成，避免权限问题
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 3. 核心逻辑：文本分析 & Prompt
# ==========================================
def analyze_text(text, target_lvl):
    raw_words = re.findall(r"[a-z]+", text.lower())
    unique_words = set(raw_words)
    
    res = []
    for w in unique_words:
        if len(w) < 2: continue
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 0)
        
        # 简单筛选逻辑：如果 rank > 0 且 rank <= target_lvl (或者没有词表时全部输出)
        if VOCAB_DICT:
            # 这里你可以自定义逻辑，例如只看 4000-8000 词
            # 为了演示，我们假设只提取“难词” (Rank > 3000)
            if rank > 3000 and rank <= target_lvl: 
                res.append((lemma, rank))
        else:
            res.append((lemma, 0))
            
    # 按词频排序 (越常见越前，或者反之)
    res.sort(key=lambda x: x[1])
    return [x[0] for x in res]

def get_ai_prompt(words):
    w_list = ", ".join(words)
    # 使用 Markdown 表格或管道符，让 AI 生成结构化数据
    # 管道符 | 比 CSV 逗号更安全，因为例句里常有逗号
    return f"""
Act as a Dictionary API. I need Anki card data for these words.
Words: {w_list}

**Strict Output Format (Pipe Separated, NO Header):**
Word | IPA | Chinese Definition | 2 English Sentences (Cn translation included) | Etymology/Root

**Requirements:**
1. Use `|` as separator.
2. Example Sentences: Use `<br>` to separate the two sentences.
3. Definition: Concise Chinese.
4. Etymology: Very short root explanation (Chinese).

**Example Line:**
benevolent | /bəˈnevələnt/ | 仁慈的 | He is benevolent.<br>She smiled benevolently. | bene(好) + vol(意愿)
"""

# ==========================================
# 4. Streamlit UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow (Cloud)")
st.caption("Step 1: 提取单词 -> Step 2: AI 生成 -> Step 3: 一键打包 iOS")

# 使用 Tab 分隔步骤，逻辑更清晰
t1, t2 = st.tabs(["1. 分析 & 提词", "2. 生成 & 下载"])

with t1:
    c1, c2 = st.columns(2)
    max_rank = c1.number_input("筛选词频上限 (Rank)", 5000, 20000, 10000, step=1000)
    
    txt = st.text_area("粘贴英文文本", height=150)
    
    if st.button("🔍 分析文本"):
        if not txt.strip():
            st.warning("请先粘贴文本")
        else:
            final_words = analyze_text(txt, max_rank)
            st.session_state['words'] = final_words
            st.success(f"筛选出 {len(final_words)} 个单词")

    if 'words' in st.session_state:
        # 允许用户二次编辑
        words_str = st.text_area("确认单词列表 (可手动增删)", ", ".join(st.session_state['words']))
        
        if st.button("📋 生成 AI Prompt"):
            final_list = [w.strip() for w in words_str.split(',') if w.strip()]
            prompt = get_ai_prompt(final_list)
            st.code(prompt, language="markdown")
            st.info("👆 复制上面代码块发给 ChatGPT/Claude/DeepSeek。然后把它的回复复制下来。")

with t2:
    st.markdown("##### 🛠️ 制作 Anki 包")
    st.markdown("<div class='instruction'>将 AI 回复的管道符格式内容 (不含 ```) 粘贴到下方：</div>", unsafe_allow_html=True)
    
    ai_response = st.text_area("粘贴 AI 回复数据", height=200, placeholder="word | ipa | def | ex | etym")
    deck_title = st.text_input("牌组名称", "My Vocab Deck")
    
    if st.button("📦 生成 .apkg (iOS 专用)"):
        if not ai_response.strip():
            st.error("内容为空")
        else:
            # 解析数据
            lines = ai_response.strip().split('\n')
            cards = []
            err_cnt = 0
            
            for line in lines:
                if "|" not in line: continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3: # 至少要有单词、音标、释义
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
                # 生成文件
                tmp_file_path = generate_anki_package(cards, deck_title)
                
                # 读取二进制数据用于下载
                with open(tmp_file_path, "rb") as f:
                    file_data = f.read()
                
                st.download_button(
                    label=f"📥 下载 {deck_title}.apkg",
                    data=file_data,
                    file_name=f"{deck_title}.apkg",
                    mime="application/octet-stream",
                    type="primary"
                )
                
                st.success(f"成功生成 {len(cards)} 张卡片！(iOS上下载后选择用Anki打开即可)")
                if err_cnt > 0:
                    st.warning(f"跳过了 {err_cnt} 行格式错误的行")
            else:
                st.error("未能识别有效数据，请检查分隔符是否为 |")