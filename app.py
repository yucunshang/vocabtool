import streamlit as st
import pandas as pd
import re
import os
import json
import time
import requests
import zipfile
import concurrent.futures
import lemminflect
import nltk
import random
import tempfile

# ==========================================
# 0. 依赖检查与初始化
# ==========================================
try:
    import PyPDF2
    import docx
    import genanki # 新增依赖
except ImportError:
    st.error("⚠️ 缺少必要依赖。请运行: pip install PyPDF2 python-docx genanki")
    st.stop()

# ==========================================
# 1. 基础 UI 配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro v2.0", page_icon="🚀")

st.markdown("""
<style>
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; }
    .block-container { padding-top: 1rem; }
    /* 移动端适配优化: 调整手机上的 Metric字体大小 */
    @media (max-width: 640px) {
        [data-testid="stMetricValue"] { font-size: 20px !important; }
    }
</style>
""", unsafe_allow_html=True)

# State 初始化
if "base_df" not in st.session_state: st.session_state.base_df = pd.DataFrame()
if "preview_card" not in st.session_state: st.session_state.preview_card = ""

# ==========================================
# 2. 数据层 (解决硬编码问题)
# ==========================================
@st.cache_data
def load_global_data():
    """
    建议：将原本代码中的 huge dicts 保存为 data/safe_names.json 和 data/global_ranks.json
    此处为了演示完整性，保留了部分核心数据结构，实际生产环境请从文件读取。
    """
    # 模拟从文件读取
    safe_names = {
        'will', 'mark', 'rose', 'lily', 'bill', 'joy', 'hope', 'grace', 'amber', 'frank', 
        'miles', 'dean', 'duke', 'king', 'prince', 'baker', 'smith', 'cook', 'brown', 'white', 
        'black', 'green', 'young', 'hall', 'wright', 'price', 'long', 'major', 'rich'
    }
    
    # 仅展示部分示例，实际请保留你原本完整的字典
    entity_ranks = {
        "china": 400, "usa": 200, "uk": 200, "apple": 1000, "google": 1000,
        "january": 400, "monday": 300, "christmas": 800
    }
    # 补全数字
    for _nw in ["one", "two", "three", "ten", "hundred", "thousand", "million"]:
        entity_ranks[_nw] = 1000
        
    return safe_names, entity_ranks

SAFE_NAMES_DB, GLOBAL_ENTITY_RANKS = load_global_data()

# 保持原有的 NLP 加载逻辑
@st.cache_data
def load_vocab_resources():
    # 这里原样保留你之前的加载逻辑，为了节省篇幅略去，请保留原代码中的 load_knowledge_base 实现
    # ... (Keep your original implementation here) ...
    return {}, {}, {}, set() # Placeholder

BUILTIN_TECHNICAL_TERMS, PROPER_NOUNS_DB, BUILTIN_PATCH_VOCAB, AMBIGUOUS_WORDS = load_vocab_resources()
NLTK_NAMES_DB = set() # Placeholder, keep original nltk logic

# 原有的 load_vocab 函数保持不变，记得把 GLOBAL_ENTITY_RANKS 传进去
vocab_dict = {} # Placeholder for vocab loading

# ==========================================
# 3. 功能函数：Prompt 优化与 Anki 生成
# ==========================================

def get_dynamic_prompt_template_v2(front_style, add_pos, def_lang, ex_count, add_ety, split_polysemy):
    """
    V2 升级：增加了 One-Shot Example (样本)，大幅提高 AI 输出稳定性
    """
    front_desc = "phrase using the word" if front_style == "phrase" else "the word itself"
    pos_instr = "append ' (pos)'" if add_pos else "no pos tag"
    
    # 构建 One-Shot 示例
    example_input = "book"
    example_output = ""
    if def_lang == "en":
        example_output = '"book (n)","A set of written or printed pages...<br><br><em>I read a good book yesterday.</em>"'
    else:
        example_output = '"book (n)","【名】书，书籍<br><br><em>I read a good book yesterday.</em>"'

    prompt = f"""# Role
Expert Linguist & Anki Card Generator.

# Task
Generate flashcards for the provided words. 
Format: CSV-style "Front","Back"

# Rules
1. Format: STRICTLY "Front_Content","Back_Content_HTML" per line.
2. Front: {front_desc}. {pos_instr}.
3. Back: Definition in {def_lang}. {f"Include {ex_count} example sentences (wrapped in <em>)." if ex_count > 0 else "NO examples."} { "Include Etymology." if add_ety else ""}
4. Output: ONLY the code block. NO explanations.

# One-Shot Example (Follow this format strictly)
Input: {example_input}
Output:
{example_output}

# Input Words:
"""
    return prompt

def generate_anki_package(cards_data, deck_name="VocabMaster Deck"):
    """
    使用 genanki 生成 .apkg 文件
    cards_data: list of tuples (front, back)
    """
    # 1. 定义样式
    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)
    
    my_model = genanki.Model(
        model_id,
        'Vocab Master Model',
        fields=[{'name': 'Front'}, {'name': 'Back'}],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '<div class="front">{{Front}}</div>',
                'afmt': '{{FrontSide}}<hr id="answer"><div class="back">{{Back}}</div>',
            },
        ],
        css='.front {font-size: 24px; font-weight: bold; text-align: center;} .back {font-size: 18px; text-align: left;}'
    )

    my_deck = genanki.Deck(deck_id, deck_name)

    for front, back in cards_data:
        # 清洗一下可能的 CSV 引号
        f_clean = front.strip('"').strip("'")
        b_clean = back.strip('"').strip("'")
        my_note = genanki.Note(model=my_model, fields=[f_clean, b_clean])
        my_deck.add_note(my_note)

    # 生成临时文件
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.apkg')
    genanki.Package(my_deck).write_to_file(tmp.name)
    return tmp.name

# ==========================================
# 4. API 交互 (支持预览)
# ==========================================
def call_deepseek_simple(prompt, api_key):
    """用于单次预览的轻量级调用"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        resp = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=20)
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 5. UI 主视图
# ==========================================
# --- 侧边栏：Secrets 管理 ---
with st.sidebar:
    st.header("🔑 配置 (Settings)")
    # 优先使用 Secrets，否则允许用户输入
    default_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    api_key_input = st.text_input("DeepSeek API Key", value=default_key, type="password")
    if not api_key_input:
        st.warning("请输入 API Key 以使用 AI 功能")

    st.divider()
    st.info("💡 提示：移动端请点击左上角箭头收起此菜单")

# --- 主界面：移动端适配的参数区 ---
st.title("🚀 Vocab Master Pro")

# 使用 Expander 收纳复杂参数，优化移动端首屏体验
with st.expander("⚙️ 过滤与提取参数 (Filter Settings)", expanded=True):
    # 移动端适配：使用 2 列而不是 5 列
    c1, c2 = st.columns(2)
    with c1: 
        current_level = st.number_input("起 (Min Level)", 0, 20000, 9000, 500)
        top_n = st.number_input("精选 Top N", 10, 500, 100, 10)
    with c2: 
        target_level = st.number_input("止 (Max Level)", 0, 20000, 15000, 500)
        min_rank_threshold = st.number_input("忽略前 N", 0, 20000, 6000, 500)

# 输入区
col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    raw_text = st.text_area("📥 粘贴文本", height=120)
with col_input2:
    uploaded_file = st.file_uploader("📂 或上传文件", type=["txt", "pdf", "docx", "epub"])

if st.button("🚀 开始分析 (Analyze)", type="primary", use_container_width=True):
    # ... (此处保留原本的文本解析与 Pandas 处理逻辑) ...
    # 假设处理完得到了 st.session_state.base_df
    pass 

# ==========================================
# 6. 结果与生成区 (包含预览与 Anki 导出)
# ==========================================
if not st.session_state.base_df.empty:
    # ... (Tabs 显示代码保持不变) ...
    
    # 假设当前在 "Top精选" Tab 下
    st.divider()
    st.markdown("#### 🤖 AI 制卡工作台")
    
    # 配置区
    ac1, ac2 = st.columns(2)
    with ac1:
        export_format = st.radio("导出格式:", ["Anki Deck (.apkg)", "CSV / TXT"], horizontal=True)
    with ac2:
        ui_def = st.radio("释义语言:", ["English", "Chinese", "Bilingual"], index=1, horizontal=True)

    # 动态 Prompt
    prompt_v2 = get_dynamic_prompt_template_v2(
        "phrase", True, "zh" if ui_def=="Chinese" else "en", 1, True, False
    )
    
    # 预览功能 (新增)
    if st.button("👁️ 预览首张卡片效果 (Preview 1 Card)"):
        if not api_key_input:
            st.error("请先配置 API Key")
        else:
            first_word = st.session_state.base_df.iloc[0]['raw']
            preview_prompt = f"{prompt_v2}{first_word}"
            with st.spinner("生成预览中..."):
                preview_res = call_deepseek_simple(preview_prompt, api_key_input)
                st.session_state.preview_card = preview_res
    
    if st.session_state.preview_card:
        st.info("预览结果 (Preview):")
        st.code(st.session_state.preview_card, language="csv")

    # 批量生成
    if st.button("⚡ 批量生成全部 (Batch Generate)", type="primary"):
        # ... (保留原本的多线程 call_deepseek_api_chunked 逻辑) ...
        # 假设 ai_result 是生成的 CSV 字符串
        
        # 结果处理：CSV vs Anki APKG
        ai_result_str = '...' # 模拟结果
        
        if "Anki" in export_format:
            # 解析 CSV 字符串为 List of Tuples
            # 简单的解析逻辑，实际建议用 csv 模块处理 quotechar
            lines = [line.split('","') for line in ai_result_str.split('\n') if '","' in line]
            if lines:
                apkg_path = generate_anki_package(lines, deck_name="Vocab Master AI")
                with open(apkg_path, "rb") as f:
                    st.download_button(
                        label="📥 下载 .apkg 文件 (直接导入 Anki)",
                        data=f,
                        file_name="vocab_master.apkg",
                        mime="application/apkg"
                    )
        else:
            st.download_button("📥 下载 CSV", ai_result_str, "cards.csv")