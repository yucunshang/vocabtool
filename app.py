import streamlit as st
import pandas as pd
import re
import os
import random
import time
import zlib
import sqlite3
import asyncio
import edge_tts
from collections import Counter
from datetime import datetime, timedelta, timezone

# 尝试导入 OpenAI，防止未安装报错
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==========================================
# 0. 页面配置 & UI 美化
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Pro", 
    page_icon="🌊", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 初始化 Session State
if 'uploader_id' not in st.session_state: st.session_state['uploader_id'] = "1000"
if 'anki_input_text' not in st.session_state: st.session_state['anki_input_text'] = ""
if 'anki_pkg_data' not in st.session_state: st.session_state['anki_pkg_data'] = None
if 'anki_pkg_name' not in st.session_state: st.session_state['anki_pkg_name'] = ""

# 全局发音人映射 (EdgeTTS)
VOICE_MAP = {
    "👩 女声 (Jenny - 推荐)": "en-US-JennyNeural",
    "👨 男声 (Christopher)": "en-US-ChristopherNeural",
    "🇬🇧 英音女 (Libby)": "en-GB-LibbyNeural",
    "🇬🇧 英音男 (Ryan)": "en-GB-RyanNeural"
}

# --- CSS 注入：美化 + 隐藏菜单 ---
st.markdown("""
<style>
    /* 1. 全局背景渐变 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 2. 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;} /* 隐藏右下角 Manage App */

    /* 3. 卡片容器样式 */
    .css-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid #eef2f6;
    }

    /* 4. 按钮样式优化 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 45px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* 5. 进度条颜色 */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    
    /* 6. 文本域字体 */
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    
    /* 7. 标题样式 */
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    
    /* 8. Radio 横向间距 */
    div[role="radiogroup"] > label { margin-right: 15px; }
    
    /* 9. Status 容器样式优化 */
    .stStatusWidget { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心工具函数 (缓存与资源加载)
# ==========================================
@st.cache_resource(show_spinner=False)
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
    import pypdf, docx, ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

def get_genanki():
    import genanki, tempfile
    return genanki, tempfile

@st.cache_data
def load_vocab_data():
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
    keys = ['gen_words_data', 'raw_count', 'process_time', 'stats_info', 'anki_pkg_data', 'anki_pkg_name', 'anki_input_text']
    for k in keys: 
        if k in st.session_state: del st.session_state[k]
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))

# ==========================================
# 2. 文本提取与分析逻辑
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    _, tempfile = get_genanki()
    text = ""
    ftype = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if ftype == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif ftype == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ftype == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ftype == 'epub':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator=' ', strip=True) + " "
            os.remove(tmp_path)
    except Exception as e: return f"Error: {e}"
    return text

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    nltk, lemminflect = load_nlp_resources()
    def get_lemma(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    valid_tokens = [t.lower() for t in tokens if len(t)>1 and not re.search(r'(.)\1{2,}', t)]
    counts = Counter(valid_tokens)
    
    candidates = []
    seen = set()
    stats = {"known":0, "target":0, "total": sum(counts.values())}
    
    for w, c in counts.items():
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, VOCAB_DICT.get(w, 99999))
        
        if rank < current_lvl: stats["known"] += c
        elif current_lvl <= rank <= target_lvl: stats["target"] += c
        
        if (current_lvl <= rank <= target_lvl) or (rank==99999 and include_unknown):
            final_w = lemma if rank != 99999 else w
            if final_w not in seen:
                candidates.append((final_w, rank))
                seen.add(final_w)
                
    candidates.sort(key=lambda x: x[1])
    return candidates, len(tokens), {
        "coverage": stats["known"]/stats["total"] if stats["total"] else 0,
        "target_density": stats["target"]/stats["total"] if stats["total"] else 0
    }

# ==========================================
# 3. AI 生成逻辑 (分批 + 实时反馈)
# ==========================================
def process_ai_in_batches(words_list, status_container):
    """
    分批调用 AI，使用 status_container 实现实时反馈
    """
    if not OpenAI:
        status_container.error("❌ 未安装 OpenAI 库，请检查 requirements.txt")
        return None
        
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        status_container.error("❌ 未配置 OPENAI_API_KEY")
        return None

    client = OpenAI(api_key=api_key, base_url=st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    
    BATCH_SIZE = 10 
    total_words = len(words_list)
    full_results = []
    
    status_container.write(f"🔌 已连接 AI 引擎，准备处理 {total_words} 个单词...")
    
    for i in range(0, total_words, BATCH_SIZE):
        batch = words_list[i : i + BATCH_SIZE]
        
        # 实时更新 Log
        status_container.write(f"🤖 正在思考第 {i+1} - {min(i+BATCH_SIZE, total_words)} 个单词...")
        
        # 优化 Prompt，明确要求中文简洁释义
        user_prompt = f"""
Task: Convert English words to Anki cards.
Format: Word ||| Chinese Definition (Concise, Simplified Chinese) ||| English Sentence (Simple, Authentic)
Rules:
1. One entry per line.
2. Definition must be in Chinese.
3. Sentence must be in English.

Input Words:
{", ".join(batch)}
"""
        try:
            resp = client.chat.completions.create(
                model=st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a vocabulary assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            full_results.append(resp.choices[0].message.content)
        except Exception as e:
            status_container.warning(f"⚠️ 第 {i//BATCH_SIZE + 1} 批次处理失败: {e}")
            
    return "\n".join(full_results)

# ==========================================
# 4. 音频并发加速逻辑 (核心加速)
# ==========================================
async def generate_audio_concurrent(text, output_file, voice, semaphore):
    """单个音频生成任务，受信号量控制"""
    async with semaphore:  # 限制并发数，防止 429 Error
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            return True
        except Exception as e:
            # 失败不中断，返回 False
            return False

async def batch_generate_audio(tasks, update_callback):
    """批量执行音频生成"""
    semaphore = asyncio.Semaphore(5) # 允许同时下载 5 个音频，速度提升明显
    
    async_tasks = []
    for t in tasks:
        async_tasks.append(generate_audio_concurrent(t['text'], t['path'], t['voice'], semaphore))
    
    # 使用 as_completed 获取实时进度
    total = len(async_tasks)
    completed = 0
    if total == 0: return

    for f in asyncio.as_completed(async_tasks):
        await f
        completed += 1
        # 每完成 2 个或最后一次时更新 UI
        if completed % 2 == 0 or completed == total:
            update_callback(f"🔊 极速合成语音中: {completed}/{total} (并发加速)")

def run_async_batch(tasks, callback):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(batch_generate_audio(tasks, callback))
    finally:
        loop.close()

def parse_anki_data(raw_text):
    cards = []
    # 清理 markdown 代码块
    text = re.sub(r'```(?:text|csv)?\s*', '', raw_text)
    text = re.sub(r'\s*```', '', text)
    
    lines = text.split('\n')
    seen = set()
    
    for line in lines:
        line = line.strip()
        if not line or "|||" not in line: continue
        
        parts = line.split("|||")
        if len(parts) >= 3:
            w = parts[0].strip()
            if w.lower() in seen: continue
            seen.add(w.lower())
            
            cards.append({
                'w': w, 
                'm': parts[1].strip(), 
                'e': parts[2].strip()
            })
    return cards

def generate_anki_package(cards_data, deck_name, enable_tts, tts_voice, status_update_func):
    genanki, tempfile = get_genanki()
    media_files = [] 
    tmp_dir = tempfile.gettempdir()
    
    # 1. 准备音频任务列表 (并发前置准备)
    audio_tasks = []
    if enable_tts:
        status_update_func("📝 正在规划音频任务列表...")
        for idx, c in enumerate(cards_data):
            phrase = c['w']
            # 简单的文件名生成，防止特殊字符
            safe_phrase = re.sub(r'[^a-zA-Z0-9]', '_', phrase)[:20]
            uid = int(time.time()*1000) + idx
            
            # 单词音频
            p_fname = f"tts_{safe_phrase}_{uid}_p.mp3"
            p_path = os.path.join(tmp_dir, p_fname)
            c['audio_p_field'] = f"[sound:{p_fname}]"
            media_files.append(p_path)
            audio_tasks.append({'text': phrase, 'path': p_path, 'voice': tts_voice})
            
            # 如需例句音频，可解开以下注释 (会增加打包时间)
            # audio_tasks.append({'text': c['e'], 'path': ..., 'voice': tts_voice})

        # 2. 并发执行音频下载
        if audio_tasks:
            run_async_batch(audio_tasks, status_update_func)

    # 3. 生成卡片 (瞬间完成)
    status_update_func("📦 正在打包 .apkg 文件...")
    
    # Anki 模板 CSS
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .meaning { font-size: 20px; font-weight: bold; color: #222; margin-bottom: 15px; }
    .example { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; color: #555; font-style: italic; font-size: 18px; text-align: left; margin-top: 15px;}
    """
    
    model = genanki.Model(
        1607392319, 'VocabFlow Pro Model',
        fields=[{'name': 'Word'}, {'name': 'Meaning'}, {'name': 'Example'}, {'name': 'Audio'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="phrase">{{Word}}</div><br>{{Audio}}',
            'afmt': '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div><div class="example">🗣️ {{Example}}</div>'
        }],
        css=CSS
    )
    deck = genanki.Deck(zlib.adler32(deck_name.encode('utf-8')), deck_name)
    
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[
            c.get('w',''), c.get('m',''), c.get('e',''), c.get('audio_p_field','')
        ]))
        
    pkg = genanki.Package(deck)
    pkg.media_files = media_files
    
    out_path = os.path.join(tmp_dir, f"{deck_name}.apkg")
    pkg.write_to_file(out_path)
    return out_path

# ==========================================
# 5. UI 主程序
# ==========================================
# 漂亮的标题栏
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>🌊 Vocab Flow <span style='color:#3498db'>Pro</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 14px; margin-bottom: 30px;'>极速单词卡片生成工作流</p>", unsafe_allow_html=True)

if not VOCAB_DICT: st.error("⚠️ 缺失词频数据文件 (coca_cleaned.csv)")

tab_extract, tab_anki = st.tabs(["📊 智能提取", "🛠️ 卡片工坊"])

# --- Tab 1: 提取与生成 ---
with tab_extract:
    # 容器1：输入区
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### 📥 文本输入")
    col1, col2 = st.columns([3, 2])
    with col1:
        # 支持文件上传
        uploaded_file = st.file_uploader("上传文件 (PDF/EPUB/DOCX)", type=['txt','pdf','docx','epub'], label_visibility="collapsed")
        raw_input_text = st.text_area("或直接粘贴文本", height=120, placeholder="在此粘贴英语文章或单词列表...")
    with col2:
        st.write("⚙️ **过滤设置**")
        min_rank = st.number_input("忽略前 N 高频", 1, 20000, 6000, step=500, help="例如：设置 6000 表示忽略最常见的 6000 个单词")
        max_rank = st.number_input("忽略后 N 低频", 2000, 50000, 15000, step=500)
        
    if st.button("🔍 开始分析", type="primary", use_container_width=True):
        input_content = extract_text_from_file(uploaded_file) if uploaded_file else raw_input_text
        if len(input_content) > 2:
            candidates, raw_c, stats = analyze_logic(input_content, min_rank, max_rank, False)
            st.session_state['gen_words_data'] = candidates
            st.session_state['raw_count'] = raw_c
            st.session_state['stats_info'] = stats
        else:
            st.toast("⚠️ 请先输入文本或上传文件", icon="🚫")
    st.markdown('</div>', unsafe_allow_html=True)

    # 结果显示区
    if st.session_state.get('gen_words_data'):
        words = [x[0] for x in st.session_state['gen_words_data']]
        
        # 容器2：统计卡片
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("文章总词数", st.session_state['raw_count'])
        c2.metric("生词密度", f"{st.session_state['stats_info']['target_density']:.1%}")
        c3.metric("提取生词", len(words))
        c4.metric("预计耗时", f"~{max(1, len(words)//10 * 3)}s") 
        
        with st.expander(f"👁️ 查看所有 {len(words)} 个生词"):
            st.text_area("可编辑列表", ", ".join(words), height=100)
        st.markdown('</div>', unsafe_allow_html=True)

        # 容器3：核心操作区
        st.markdown("### 🚀 生成与下载")
        
        # 语音选择 (解决单调 UI)
        col_voice, col_check = st.columns([3, 1])
        with col_voice:
            sel_voice = st.radio("🎙️ AI 发音人 (Neural TTS)", list(VOICE_MAP.keys()), horizontal=True)
        with col_check:
            st.write("") 
            enable_audio = st.checkbox("启用语音", value=True, help="并发加速已开启，速度极快")

        col_btn, col_down = st.columns([1, 1])
        
        with col_btn:
            if st.button("✨ 一键生成 Anki 包", type="primary", use_container_width=True):
                # 解决反馈慢：使用 st.status
                status_box = st.status("🚀 初始化任务...", expanded=True)
                
                # 1. 限制数量 (防止过慢)
                MAX_W = 300
                if len(words) > MAX_W:
                    status_box.warning(f"⚠️ 单词较多，自动截取前 {MAX_W} 个处理")
                target_w = words[:MAX_W]
                
                # 2. AI 处理
                ai_text = process_ai_in_batches(target_w, status_box)
                
                if ai_text:
                    st.session_state['anki_input_text'] = ai_text
                    
                    # 3. 打包与语音 (UI 回调函数)
                    def update_status_label(msg):
                        status_box.update(label=msg)
                        status_box.write(msg) # 同时也写入 log
                    
                    try:
                        deck_name = f"Vocab_{get_beijing_time_str()}"
                        f_path = generate_anki_package(
                            parse_anki_data(ai_text), 
                            deck_name, 
                            enable_audio, 
                            VOICE_MAP[sel_voice], 
                            update_status_label
                        )
                        
                        with open(f_path, "rb") as f:
                            st.session_state['anki_pkg_data'] = f.read()
                        st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"
                        
                        status_box.update(label="✅ 所有任务完成！", state="complete", expanded=False)
                        st.balloons()
                    except Exception as e:
                        status_box.update(label="❌ 发生错误", state="error")
                        st.error(str(e))
                else:
                    status_box.update(label="❌ AI 生成失败", state="error")
        
        # 下载按钮
        with col_down:
            if st.session_state.get('anki_pkg_data'):
                st.download_button(
                    label=f"📥 下载 {st.session_state['anki_pkg_name']}",
                    data=st.session_state['anki_pkg_data'],
                    file_name=st.session_state['anki_pkg_name'],
                    mime="application/octet-stream",
                    type="secondary",
                    use_container_width=True
                )
            else:
                st.info("👈 点击左侧按钮开始生成")

        # 提示信息 (解决痛点：说明第三方 AI)
        st.markdown("---")
        st.info("💡 **追求极致释义质量？**\n\n内置 AI 追求速度与便利。如需 GPT-4/Claude 的深度释义，请复制下方 Prompt 到官网生成，然后在 **Tab 2** 导入。", icon="🧠")
        with st.expander("📌 复制第三方 AI Prompt"):
            st.code(f"""Task: Create Anki cards.
Input: {", ".join(words[:50])}...
Format: Word ||| Chinese Meaning ||| English Example
Rules: One entry per line. Concise Chinese.""", language="text")

# --- Tab 2: 手动模式 ---
with tab_anki:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.write("📦 **手动导入模式** (适用于第三方 AI 结果)")
    
    txt_input = st.text_area("粘贴内容 (格式: Word ||| Meaning ||| Example)", height=200, placeholder="hectic ||| 繁忙的 ||| It was a hectic day.")
    
    col_m_voice, col_m_act = st.columns([3, 1])
    with col_m_voice:
        m_voice_key = st.radio("语音选择", list(VOICE_MAP.keys()), horizontal=True, key="manual_voice")
    with col_m_act:
        st.write("")
        btn_manual = st.button("🛠️ 生成卡片", use_container_width=True)
        
    if btn_manual:
        if not txt_input.strip():
            st.warning("⚠️ 内容为空")
        else:
            status_m = st.status("处理中...", expanded=True)
            try:
                deck_name = f"Manual_{get_beijing_time_str()}"
                
                def update_m(m): status_m.write(m)
                
                f_path = generate_anki_package(
                    parse_anki_data(txt_input), 
                    deck_name, 
                    True, 
                    VOICE_MAP[m_voice_key], 
                    update_m
                )
                with open(f_path, "rb") as f:
                    st.download_button("📥 下载生成的文件", f, file_name=f"{deck_name}.apkg", type="primary", use_container_width=True)
                status_m.update(label="✅ 完成", state="complete")
            except Exception as e:
                status_m.update(label="❌ 错误", state="error")
                st.error(e)
            
    st.markdown('</div>', unsafe_allow_html=True)