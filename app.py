"""
Vocab Flow Ultra – Streamlit entry point.
UI and session-state wiring; logic lives in constants, errors, utils,
resources, extraction, vocab, ai, anki_parse, tts, anki_package, state.
"""
import html
import logging
import os
import re
import time

import pandas as pd
import streamlit as st

import constants
import resources
from ai import CardFormat, build_card_prompt, get_word_quick_definition, process_ai_in_batches
from anki_package import cleanup_old_apkg_files, generate_anki_package
from anki_parse import parse_anki_data
from config import get_config
from extraction import (
    extract_text_from_file,
    extract_text_from_url,
    is_upload_too_large,
    parse_anki_txt_export,
)
from state import clear_all_state, set_generated_words_state
from utils import get_beijing_time_str, render_copy_button, run_gc
from vocab import analyze_logic

# Load vocab and expose to app (and to resources for vocab/ai modules)
VOCAB_DICT, FULL_DF = resources.load_vocab_data()
resources.VOCAB_DICT = VOCAB_DICT
resources.FULL_DF = FULL_DF

# Clean old .apkg files from our temp subdir (e.g. from previous sessions)
cleanup_old_apkg_files()

# Stop words to filter out in direct-input mode (articles, pronouns,
# prepositions, conjunctions, auxiliary verbs, determiners, etc.).
_DIRECT_INPUT_STOPWORDS: set = {
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "out", "off", "into", "onto", "upon", "about", "over",
    "under", "after", "before", "between", "through", "during",
    "above", "below", "around", "against", "along", "across",
    "behind", "beyond", "within", "without", "toward", "towards",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "for",
    "both", "either", "neither", "whether",
    # Auxiliary / common verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "did", "does", "done", "doing",
    "has", "had", "have", "having",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    # Very common adverbs / particles
    "not", "no", "yes", "very", "too", "also", "just",
    "then", "than", "now", "here", "there",
    "how", "when", "where", "why",
    # Other function words
    "if", "as", "all", "each", "every", "any", "some",
    "such", "more", "most", "much", "many", "few",
    "other", "own", "same", "only",
}

logger = logging.getLogger(__name__)

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra",
    page_icon="⚡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
for key, default_value in constants.DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Handle clickable word lookup via query params
_qp = st.query_params
_clicked_word = _qp.get("lookup_word", "")
if _clicked_word:
    st.query_params.clear()
    st.session_state["quick_lookup_word"] = _clicked_word
    st.session_state["_auto_lookup_word"] = _clicked_word

# Custom CSS – app-like design
st.markdown("""
<style>
    /* ===== Global: hide Streamlit chrome, set base font ===== */
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
                     'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
    }

    /* ===== Smooth transitions on all interactive elements ===== */
    button, input, textarea, [data-baseweb="tab"], .stExpander {
        transition: all 0.2s ease !important;
    }

    /* ===== Buttons: pill-shaped, elevated feel ===== */
    .stButton>button {
        border-radius: 10px; font-weight: 600; width: 100%; margin-top: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        letter-spacing: 0.01em;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }
    .stButton>button:active { transform: translateY(0); }

    /* ===== Text areas ===== */
    .stTextArea textarea {
        font-family: 'Consolas', 'SF Mono', 'Monaco', monospace;
        font-size: 14px; border-radius: 10px;
    }

    /* ===== Form cards ===== */
    .stForm {
        border: 1px solid #e5e7eb; border-radius: 14px;
        padding: 1.25rem 1.5rem; background: #fafbfc;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    /* ===== Metric cards ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1rem; border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700; letter-spacing: -0.02em;
    }

    /* ===== Tabs: segmented-control style ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #f1f5f9; padding: 4px;
        border-radius: 12px; border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 1rem; border-radius: 10px;
        font-weight: 500; font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* ===== Expanders ===== */
    .stExpander {
        border: 1px solid #e5e7eb; border-radius: 12px;
        margin-bottom: 10px; overflow: hidden;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* ===== Progress bar ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 6px;
    }

    /* ===== Section dividers ===== */
    hr { border: none; height: 1px; background: #e5e7eb; margin: 1.5rem 0; }

    /* ===== App footer ===== */
    .app-footer {
        margin-top: 3rem; padding: 1.25rem 0; text-align: center;
        color: #94a3b8; font-size: 0.8rem; letter-spacing: 0.02em;
        border-top: 1px solid #f1f5f9;
    }

    /* ===== Hero header ===== */
    .app-hero {
        text-align: center; padding: 1.5rem 0 0.5rem;
    }
    .app-hero h1 {
        font-size: 1.75rem; font-weight: 800; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.25rem;
    }
    .app-hero p {
        color: #64748b; font-size: 0.9rem; margin: 0;
    }

    /* ===== Radio buttons: chip style ===== */
    .stRadio > div { gap: 0.4rem; }
    .stRadio > div > label {
        border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 0.3rem 0.75rem; font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    .stRadio > div > label:hover {
        border-color: #93c5fd; background: #f0f9ff;
    }

    /* ===== Number inputs ===== */
    .stNumberInput input { border-radius: 10px; }

    /* ===== Toast / info / warning boxes ===== */
    .stAlert { border-radius: 10px; }

    /* ===== Download button ===== */
    .stDownloadButton > button {
        border-radius: 10px; font-weight: 600;
        box-shadow: 0 2px 8px rgba(59,130,246,0.15);
    }
</style>
""", unsafe_allow_html=True)


def set_anki_pkg(file_path: str, deck_name: str) -> None:
    """Store Anki package path in session state and clean previous file."""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError("Generated Anki package file not found.")

    prev_path = st.session_state.get('anki_pkg_path')
    if prev_path and prev_path != file_path:
        try:
            if os.path.exists(prev_path):
                os.remove(prev_path)
        except OSError as e:
            logger.warning("Could not remove previous anki package: %s", e)

    st.session_state['anki_pkg_path'] = file_path
    st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"


def render_anki_download_button(
    label: str,
    *,
    button_type: str = "primary",
    use_container_width: bool = False
) -> None:
    """Safely render Anki package download button if file exists."""
    file_path = st.session_state.get('anki_pkg_path')
    file_name = st.session_state.get('anki_pkg_name', "deck.apkg")

    if not file_path:
        return
    if not os.path.exists(file_path):
        st.warning("⚠️ 下载文件不存在，请重新生成。")
        st.session_state['anki_pkg_path'] = ""
        return

    try:
        with open(file_path, "rb") as f:
            st.download_button(
                label=label,
                data=f.read(),
                file_name=file_name,
                mime="application/octet-stream",
                type=button_type,
                use_container_width=use_container_width
            )
    except OSError as e:
        logger.error("Failed to open package for download: %s", e)
        st.error("❌ 下载文件读取失败，请重新生成。")


# ==========================================
# UI Components
# ==========================================


def render_card_format_selector(key_prefix: str) -> CardFormat:
    """Render card format options and return the selected CardFormat dict.

    ``key_prefix`` makes widget keys unique when called in multiple places.
    """
    st.markdown("#### ⚙️ 卡片格式自定义")

    col_front, col_def = st.columns(2)
    with col_front:
        front_label = st.radio(
            "正面内容",
            options=list(constants.FRONT_OPTIONS.keys()),
            index=1,
            horizontal=True,
            key=f"{key_prefix}_front",
        )
    with col_def:
        def_label = st.radio(
            "释义语言",
            options=list(constants.DEFINITION_OPTIONS.keys()),
            index=0,
            horizontal=True,
            key=f"{key_prefix}_def",
        )

    col_ex, col_ety = st.columns(2)
    with col_ex:
        ex_label = st.radio(
            "例句数量",
            options=list(constants.EXAMPLE_COUNT_OPTIONS.keys()),
            index=0,
            horizontal=True,
            key=f"{key_prefix}_ex",
        )
    with col_ety:
        ety_label = st.radio(
            "词源词根",
            options=list(constants.ETYMOLOGY_OPTIONS.keys()),
            index=0,
            horizontal=True,
            key=f"{key_prefix}_ety",
        )

    return CardFormat(
        front=constants.FRONT_OPTIONS[front_label],
        definition=constants.DEFINITION_OPTIONS[def_label],
        examples=constants.EXAMPLE_COUNT_OPTIONS[ex_label],
        etymology=constants.ETYMOLOGY_OPTIONS[ety_label],
    )


st.markdown("""
<div class="app-hero">
    <h1>Vocab Flow Ultra</h1>
    <p>文本 → 词表 → Anki 牌组，一步到位 · AI 释义 · 词源拆解 · 并发语音</p>
</div>
""", unsafe_allow_html=True)


def _do_lookup(query_word: str) -> None:
    """Execute AI lookup for a word, populating session state cache and result."""
    st.session_state["quick_lookup_is_loading"] = True
    try:
        cache_key = f"lookup_cache_{query_word.lower()}"
        if cache_key not in st.session_state:
            with st.spinner("🔍 查询中..."):
                st.session_state[cache_key] = get_word_quick_definition(query_word)
            keys = st.session_state["quick_lookup_cache_keys"]
            keys.append(cache_key)
            while len(keys) > constants.QUICK_LOOKUP_CACHE_MAX:
                old_key = keys.pop(0)
                if old_key in st.session_state:
                    del st.session_state[old_key]
            st.session_state["quick_lookup_cache_keys"] = keys
        st.session_state["quick_lookup_last_query"] = query_word
        st.session_state["quick_lookup_last_result"] = st.session_state.get(cache_key)
    finally:
        st.session_state["quick_lookup_is_loading"] = False
        st.session_state["quick_lookup_block_until"] = time.time() + constants.QUICK_LOOKUP_COOLDOWN_SECONDS


def render_quick_lookup() -> None:
    st.markdown("### AI 极速查词")
    st.caption("输入单词后按回车或点击查询 · 释义中英文单词可点击继续查询")

    if "quick_lookup_last_query" not in st.session_state:
        st.session_state["quick_lookup_last_query"] = ""
    if "quick_lookup_last_result" not in st.session_state:
        st.session_state["quick_lookup_last_result"] = None
    if "quick_lookup_is_loading" not in st.session_state:
        st.session_state["quick_lookup_is_loading"] = False
    if "quick_lookup_block_until" not in st.session_state:
        st.session_state["quick_lookup_block_until"] = 0.0
    if "quick_lookup_cache_keys" not in st.session_state:
        st.session_state["quick_lookup_cache_keys"] = []

    now_ts = time.time()
    in_cooldown = now_ts < st.session_state["quick_lookup_block_until"]
    lookup_disabled = st.session_state["quick_lookup_is_loading"] or in_cooldown

    # Apply pending lookup word before text_input widget is created.
    # This avoids mutating an already-instantiated widget state key.
    pending_word = st.session_state.pop("_quick_lookup_pending_word", "")
    if pending_word:
        st.session_state["quick_lookup_word"] = pending_word
        st.session_state["_auto_lookup_word"] = pending_word

    # Auto-lookup from clicked word (query param, pills, or word-block)
    auto_word = st.session_state.pop("_auto_lookup_word", "")
    if auto_word and not in_cooldown:
        _do_lookup(auto_word)

    with st.form("quick_lookup_form", clear_on_submit=False):
        col_word, col_btn = st.columns([4, 1])
        with col_word:
            lookup_word = st.text_input(
                "输入单词或短语",
                placeholder="如：serendipity, take off, run into...",
                key="quick_lookup_word",
                label_visibility="collapsed",
                autocomplete="off",
            )
        with col_btn:
            lookup_submit = st.form_submit_button(
                "查询中..." if st.session_state["quick_lookup_is_loading"] else "查询",
                type="primary",
                use_container_width=True,
                disabled=lookup_disabled
            )

    if in_cooldown:
        wait_seconds = max(0.0, st.session_state["quick_lookup_block_until"] - now_ts)
        st.caption(f"⏱️ 请稍候 {wait_seconds:.1f}s 再次查询")

    if lookup_submit:
        query_word = lookup_word.strip()
        if not query_word:
            st.warning("⚠️ 请输入单词或短语。")
        else:
            if st.session_state["quick_lookup_is_loading"]:
                st.info("⏳ 查询进行中，请稍候。")
            elif time.time() < st.session_state["quick_lookup_block_until"]:
                st.info("⏱️ 请求过于频繁，请稍后再试。")
            else:
                _do_lookup(query_word)

    result = st.session_state.get("quick_lookup_last_result")
    if result and 'error' not in result:
        raw_content = result['result']
        rank = result.get('rank', 99999)

        if rank <= 5000:
            rank_color = "#10b981"
            rank_label = "高频词"
        elif rank <= 10000:
            rank_color = "#3b82f6"
            rank_label = "常用词"
        elif rank <= 20000:
            rank_color = "#f59e0b"
            rank_label = "进阶词"
        elif rank < 99999:
            rank_color = "#ef4444"
            rank_label = "专业词"
        else:
            rank_color = "#6b7280"
            rank_label = "未收录"

        # Build styled HTML lines (no iframe needed)
        lines = raw_content.split('\n')
        formatted_lines = []
        clickable_words: list[str] = []

        current_query = st.session_state.get("quick_lookup_last_query", "").lower().strip()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Collect English words from all result lines so users can
            # continue lookup from definitions, etymology, and examples.
            for w in re.findall(r"[a-zA-Z]{3,}", line):
                wl = w.lower()
                if wl == current_query:
                    continue
                if wl not in _DIRECT_INPUT_STOPWORDS and wl not in clickable_words:
                    clickable_words.append(wl)

            if line.startswith("🌱"):
                safe = html.escape(line)
                formatted_lines.append(f'<div style="color:#065f46;background:#ecfdf5;padding:6px 10px;border-radius:8px;margin:8px 0;line-height:1.7;">{safe}</div>')
            elif "|" in line and len(line) < 50:
                safe = html.escape(line)
                formatted_lines.append(f'<div style="color:#1e3a8a;margin-bottom:6px;font-size:16px;line-height:1.7;">{safe}</div>')
            elif line.startswith("•"):
                safe = html.escape(line)
                formatted_lines.append(f'<div style="color:#374151;margin-top:6px;font-size:16px;line-height:1.7;">{safe}</div>')
            else:
                safe = html.escape(line)
                formatted_lines.append(f'<div style="color:#6b7280;margin-bottom:8px;font-size:16px;line-height:1.7;">{safe}</div>')

        display_html = "".join(formatted_lines)
        rank_badge = f'<span style="display:inline-block;background:{rank_color};color:white;padding:3px 10px;border-radius:5px;font-size:13px;font-weight:600;">📊 {rank} · {rank_label}</span>'

        st.markdown(f"""<div style="padding:4px 0;">{display_html}<div style="margin-top:10px;">{rank_badge}</div></div>""", unsafe_allow_html=True)

        # Clickable word pills for continuing lookup (pure Streamlit, no iframe)
        if clickable_words:
            picked = st.pills(
                "点击单词继续查询",
                options=clickable_words[:20],
                key="ql_word_pills",
                label_visibility="collapsed",
            )
            if "ql_word_pills_last" not in st.session_state:
                st.session_state["ql_word_pills_last"] = ""
            if picked and picked != st.session_state["ql_word_pills_last"]:
                st.session_state["ql_word_pills_last"] = picked
                st.session_state["_quick_lookup_pending_word"] = picked
                st.rerun()

    elif result and 'error' in result:
        st.error(f"❌ 查询失败：{result.get('error', '未知错误')}")

    st.markdown("---")


if hasattr(st, "fragment"):
    render_quick_lookup = st.fragment(render_quick_lookup)

render_quick_lookup()

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv` 或 `vocab.pkl` 文件，请检查目录。")

with st.expander("使用指南 & 支持格式", expanded=False):
    st.markdown("""
    **极速工作流**
    1. **查词** — 顶部 AI 查词，秒速获取精准释义、词源拆解和双语例句
    2. **提取** — 支持 PDF / ePub / Docx / TXT / CSV / Excel 等格式
    3. **生成** — AI 释义 + 并发语音合成，一键打包下载

    **支持的文件格式**
    TXT · PDF · DOCX · EPUB · CSV · XLSX · XLS · DB · SQLite · Anki 导出 (.txt)
    """)

tab_extract, tab_anki = st.tabs([
    "单词提取",
    "卡片制作"
])

# ==========================================
# Tab 1: Word Extraction
# ==========================================
with tab_extract:
    mode_context, mode_direct, mode_rank = st.tabs([
        "语境分析",
        "直接输入",
        "词频列表"
    ])

    with mode_context:
        col1, col2 = st.columns(2)
        current_rank = col1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100)
        target_rank = col2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500)

        if target_rank < current_rank:
            st.warning("⚠️ Max Rank 必须大于等于 Min Rank")

        st.markdown("#### 📥 导入内容")

        input_url = st.text_input(
            "🔗 输入文章 URL (自动抓取)",
            placeholder="https://www.economist.com/...",
            key="url_input_key"
        )

        uploaded_file = st.file_uploader(
            "或直接上传文件",
            type=['txt', 'pdf', 'docx', 'epub', 'csv', 'xlsx', 'xls', 'db', 'sqlite'],
            key=st.session_state['uploader_id'],
            label_visibility="collapsed"
        )
        if uploaded_file and is_upload_too_large(uploaded_file):
            st.error(f"❌ 文件过大，已限制为 {constants.MAX_UPLOAD_MB}MB。请缩小文件后重试。")
            uploaded_file = None

        pasted_text = st.text_area(
            "或在此粘贴文本",
            height=100,
            key="paste_key",
            placeholder="支持直接粘贴文章内容..."
        )

        if st.button("🚀 开始分析", type="primary"):
            if target_rank < current_rank:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请修正后重试。")
            else:
                with st.status("🔍 正在加载资源并分析文本...", expanded=True) as status:
                    start_time = time.time()
                    raw_text = ""

                    if input_url:
                        status.write(f"🌐 正在抓取 URL: {input_url}...")
                        raw_text = extract_text_from_url(input_url)
                    elif uploaded_file:
                        raw_text = extract_text_from_file(uploaded_file)
                    else:
                        raw_text = pasted_text

                    if len(raw_text) > 2:
                        status.write("🧠 正在进行 NLP 词形还原与分级...")
                        final_data, raw_count, stats_info = analyze_logic(
                            raw_text, current_rank, target_rank, False
                        )

                        set_generated_words_state(final_data, raw_count, stats_info)
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        status.update(label="✅ 分析完成", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 内容为空或太短", state="error")

    with mode_direct:
        st.markdown("#### 📤 导入 Anki 牌组导出文件 (可选)")
        st.caption("💡 提示：在 Anki 导出时，推荐选择 **'Notes in Plain Text'** (笔记纯文本)。但如果您选择了 **'Cards in Plain Text'**，系统也会尝试自动解析。")

        anki_export_file = st.file_uploader(
            "上传 Anki 导出的 .txt 文件",
            type=['txt'],
            key="anki_import_uploader"
        )
        if anki_export_file and is_upload_too_large(anki_export_file):
            st.error(f"❌ 文件过大，已限制为 {constants.MAX_UPLOAD_MB}MB。请缩小文件后重试。")
            anki_export_file = None

        prefilled_text = ""
        if anki_export_file:
            with st.spinner("正在智能解析 Anki 导出文件..."):
                prefilled_text = parse_anki_txt_export(anki_export_file)
                if prefilled_text:
                    st.success(f"✅ 成功提取 {len(prefilled_text.splitlines())} 个单词")

        raw_input = st.text_area(
            "✍️ 粘贴单词列表 (每行一个 或 逗号分隔)",
            height=200,
            value=prefilled_text,
            placeholder="altruism\nhectic\nserendipity"
        )

        if st.button("🚀 生成列表", key="btn_direct", type="primary"):
            with st.spinner("正在解析列表..."):
                if raw_input.strip():
                    words = [w.strip() for w in re.split(r'[,\n\t]+', raw_input) if w.strip()]
                    unique_words = []
                    seen = set()

                    for word in words:
                        w_lower = word.lower().strip()
                        if not w_lower or w_lower in seen:
                            continue
                        # Skip non-alphabetic tokens
                        if not re.match(r'^[a-zA-Z]+(?:[-\' ][a-zA-Z]+)*$', w_lower):
                            continue
                        # Skip single characters and very short stop words
                        if len(w_lower) <= 1:
                            continue
                        # Skip common stop words / function words
                        if w_lower in _DIRECT_INPUT_STOPWORDS:
                            continue
                        seen.add(w_lower)
                        unique_words.append(word)

                    raw_count = len(words)
                    data_list = [(w, VOCAB_DICT.get(w.lower(), 99999)) for w in unique_words]
                    set_generated_words_state(data_list, raw_count, None)
                    filtered = raw_count - len(unique_words)
                    msg = f"✅ 已加载 {len(unique_words)} 个单词"
                    if filtered > 0:
                        msg += f"（已过滤 {filtered} 个无关词）"
                    st.toast(msg, icon="🎉")
                else:
                    st.warning("⚠️ 内容为空。")

    with mode_rank:
        gen_type = st.radio("生成模式", ["🔢 顺序生成", "🔀 随机抽取"], horizontal=True)

        if "顺序生成" in gen_type:
            col_a, col_b = st.columns(2)
            start_rank = col_a.number_input("起始排名", 1, 20000, 8000, step=100)
            count = col_b.number_input("数量", 10, 5000, 10, step=10)

            if st.button("🚀 生成列表"):
                with st.spinner("正在提取..."):
                    if FULL_DF is not None:
                        rank_col = next(c for c in FULL_DF.columns if 'rank' in c)
                        word_col = next(c for c in FULL_DF.columns if 'word' in c)
                        subset = FULL_DF[FULL_DF[rank_col] >= start_rank].sort_values(rank_col).head(count)
                        set_generated_words_state(
                            list(zip(subset[word_col], subset[rank_col])),
                            0,
                            None
                        )
        else:
            col_min, col_max, col_cnt = st.columns([1, 1, 1])
            min_rank = col_min.number_input("最小排名", 1, 20000, 12000, step=100)
            max_rank = col_max.number_input("最大排名", 1, 25000, 15000, step=100)
            random_count = col_cnt.number_input("抽取数量", 10, 5000, 10, step=10)

            if max_rank < min_rank:
                st.warning("⚠️ 最大排名必须大于等于最小排名")

            if st.button("🎲 随机抽取"):
                if max_rank < min_rank:
                    st.error("❌ 最大排名必须大于等于最小排名，请修正后重试。")
                else:
                    with st.spinner("正在抽取..."):
                        if FULL_DF is not None:
                            rank_col = next(c for c in FULL_DF.columns if 'rank' in c)
                            word_col = next(c for c in FULL_DF.columns if 'word' in c)
                            pool = FULL_DF[(FULL_DF[rank_col] >= min_rank) & (FULL_DF[rank_col] <= max_rank)]
                            if len(pool) < random_count:
                                st.warning(f"⚠️ 该范围只有 {len(pool)} 个单词，已全部选中")
                            sample = pool.sample(n=min(random_count, len(pool)))
                            set_generated_words_state(
                                list(zip(sample[word_col], sample[rank_col])),
                                0,
                                None
                            )

    # Display results (shared across all modes)
    if st.session_state.get('gen_words_data'):
        data = st.session_state['gen_words_data']
        original_count = len(data)

        if st.session_state.get('stats_info'):
            stats = st.session_state['stats_info']
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("📊 词汇覆盖率", f"{stats['coverage']*100:.1f}%")
            with col_s2:
                st.metric("🎯 目标词密度", f"{stats['target_density']*100:.1f}%")

        raw_count = st.session_state.get('raw_count', 0)
        if not raw_count:
            raw_count = original_count
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("📦 提取的单词总数", raw_count)
        with col_t2:
            st.metric("✅ 筛选后单词总数", original_count)

        st.markdown(f"### ✅ 提取成功！")

        words_only = [w for w, r in data]
        words_text = "\n".join(words_only)
        if "word_list_editor" not in st.session_state:
            st.session_state["word_list_editor"] = words_text

        col_title, col_copy_btn = st.columns([5, 1])
        with col_title:
            st.markdown("### 📝 单词列表")
        with col_copy_btn:
            current_words_text = st.session_state.get("word_list_editor", words_text)
            render_copy_button(current_words_text, key="copy_words_btn")
        st.caption("💡 可以在下方文本框中编辑、新增或删除单词，每行一个单词")

        edited_words = st.text_area(
            f"✍️ 单词列表 (共 {original_count} 个)",
            height=300,
            key="word_list_editor",
            label_visibility="collapsed",
            help="每行一个单词"
        )

        if edited_words != words_text:
            edited_word_list = [w.strip() for w in edited_words.split('\n') if w.strip()]
            st.info(f"📝 已编辑：当前共 {len(edited_word_list)} 个单词")
            words_only = edited_word_list
        else:
            words_only = [w for w, r in data]

        st.markdown("---")
        st.markdown("### 🤖 AI 生成 Anki 卡片")

        col_ai_btn, col_copy_hint = st.columns([1, 1.35], vertical_alignment="top")

        with col_ai_btn:
            ai_model_label = get_config()["openai_model"]

            selected_voice_label = st.radio(
                "🎙️ 发音人",
                options=list(constants.VOICE_MAP.keys()),
                index=0,
                horizontal=False,
                key="sel_voice_auto"
            )
            selected_voice_code = constants.VOICE_MAP[selected_voice_label]

            enable_audio_auto = st.checkbox("启用语音", value=True, key="chk_audio_auto")

            # Keep full list for third-party prompt; only cap built-in AI path.
            words_for_auto_ai = words_only
            current_word_count = len(words_for_auto_ai)
            if current_word_count > constants.MAX_AUTO_LIMIT:
                st.caption(
                    f"⚠️ 当前 {current_word_count} 词；内置 AI 最多处理前 {constants.MAX_AUTO_LIMIT} 词。"
                    " 如需全部处理，请使用右侧第三方 Prompt 分批。"
                )
                words_for_auto_ai = words_for_auto_ai[:constants.MAX_AUTO_LIMIT]

            if st.button(f"🚀 使用 {ai_model_label} 生成", type="primary", use_container_width=True):
                progress_title = st.empty()
                stage_text = st.empty()
                overall_bar = st.progress(0.0)
                ai_bar = st.progress(0.0)
                ai_text = st.empty()
                pkg_bar = st.progress(0.0)
                pkg_text = st.empty()

                def render_stages(ai_status: str, parse_status: str, pkg_status: str) -> None:
                    stage_text.markdown(
                        f"**流程进度**  \n"
                        f"1) AI 批量生成：{ai_status}  \n"
                        f"2) 结果解析：{parse_status}  \n"
                        f"3) 打包/语音：{pkg_status}"
                    )

                progress_title.markdown("#### ⏳ 内置 AI 制卡进度")
                render_stages("进行中", "等待中", "等待中")
                ai_text.text("AI 生成：准备中...")
                pkg_text.text("打包/语音：等待中...")

                def update_ai_progress(current: int, total: int) -> None:
                    ratio = current / total if total > 0 else 0.0
                    ai_bar.progress(ratio)
                    overall_bar.progress(min(0.70, ratio * 0.70))
                    ai_text.text(f"AI 生成：已处理 {current}/{total}")

                ai_result = process_ai_in_batches(
                    words_for_auto_ai,
                    progress_callback=update_ai_progress,
                )

                if ai_result:
                    ai_bar.progress(1.0)
                    overall_bar.progress(0.75)
                    render_stages("完成", "进行中", "等待中")
                    ai_text.text("AI 生成：完成")
                    pkg_text.text("打包/语音：等待中...")
                    parsed_data = parse_anki_data(ai_result)

                    if parsed_data:
                        try:
                            overall_bar.progress(0.80)
                            render_stages("完成", "完成", "进行中")
                            pkg_text.text("打包/语音：正在生成 Anki 包...")
                            deck_name = f"Vocab_{get_beijing_time_str()}"

                            def update_pkg_progress(ratio: float, text: str) -> None:
                                pkg_bar.progress(ratio)
                                overall_bar.progress(min(1.0, 0.80 + ratio * 0.20))
                                pkg_text.text(f"打包/语音：{text}")

                            file_path = generate_anki_package(
                                parsed_data,
                                deck_name,
                                enable_tts=enable_audio_auto,
                                tts_voice=selected_voice_code,
                                progress_callback=update_pkg_progress
                            )

                            set_anki_pkg(file_path, deck_name)

                            pkg_bar.progress(1.0)
                            overall_bar.progress(1.0)
                            render_stages("完成", "完成", "完成")
                            pkg_text.markdown(f"✅ **处理完成！共生成 {len(parsed_data)} 张卡片**")
                            st.balloons()
                            run_gc()
                        except Exception as e:
                            render_stages("完成", "完成", "失败")
                            from errors import ErrorHandler
                            ErrorHandler.handle(e, "生成出错")
                    else:
                        render_stages("完成", "失败", "未开始")
                        st.error("解析失败，AI 返回内容为空或格式错误。")
                else:
                    render_stages("失败", "未开始", "未开始")
                    st.error("AI 生成失败，请检查 API Key 或网络连接。")

            render_anki_download_button(
                f"📥 下载 {st.session_state.get('anki_pkg_name', 'deck.apkg')}",
                button_type="primary",
                use_container_width=True
            )
            st.caption("⚠️ AI 结果请人工复核后再学习。")

        with col_copy_hint:
            st.markdown("#### 第三方 AI Prompt")
            st.caption("内置 AI 适合快速生成；需要更大批量时，使用下方 Prompt 到第三方 AI。")

            with st.expander("📌 复制 Prompt（第三方 AI）", expanded=False):
                card_fmt = render_card_format_selector("tab1_prompt")
                batch_size_prompt = int(
                    st.number_input("🔢 分组大小 (最大 500)", min_value=1, max_value=500, value=50, step=10)
                )
                current_batch_words = []

                if words_only:
                    total_w = len(words_only)
                    if total_w <= 500:
                        st.caption(f"💡 当前共 {total_w} 个单词（≤500），已全部放入一个 Prompt。")
                        current_batch_words = words_only
                    else:
                        num_batches = (total_w + batch_size_prompt - 1) // batch_size_prompt
                        batch_options = [
                            f"第 {i+1} 组 ({i*batch_size_prompt+1} - {min((i+1)*batch_size_prompt, total_w)})"
                            for i in range(num_batches)
                        ]
                        selected_batch_str = st.selectbox("📂 选择当前分组", batch_options)
                        sel_idx = batch_options.index(selected_batch_str)
                        current_batch_words = words_only[
                            sel_idx*batch_size_prompt:min((sel_idx+1)*batch_size_prompt, total_w)
                        ]
                else:
                    st.warning("⚠️ 暂无单词数据，请先提取单词。")

                words_str_for_prompt = ", ".join(current_batch_words) if current_batch_words else "[INSERT YOUR WORD LIST HERE]"
                strict_prompt_template = build_card_prompt(words_str_for_prompt, card_fmt)
                st.code(strict_prompt_template, language="text")

# ==========================================
# Tab 2: Manual Anki Card Creation
# ==========================================
with tab_anki:
    st.markdown("### 📦 手动制作 Anki 牌组")

    if 'anki_cards_cache' not in st.session_state:
        st.session_state['anki_cards_cache'] = None

    def reset_anki_state() -> None:
        st.session_state['anki_cards_cache'] = None
        if st.session_state.get('anki_pkg_path'):
            try:
                if os.path.exists(st.session_state['anki_pkg_path']):
                    os.remove(st.session_state['anki_pkg_path'])
            except OSError as e:
                logger.warning("Could not remove temp anki package: %s", e)
        st.session_state['anki_pkg_path'] = ""
        st.session_state['anki_pkg_name'] = ""
        st.session_state['anki_input_text'] = ""

    beijing_time_str = get_beijing_time_str()
    deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{beijing_time_str}")

    ai_response = st.text_area(
        "粘贴 AI 返回内容",
        height=300,
        key="anki_input_text",
        placeholder='hectic ||| 忙乱的 ||| She has a hectic schedule today.',
    )

    manual_voice_label = st.radio(
        "🎙️ 发音人",
        options=list(constants.VOICE_MAP.keys()),
        index=0,
        horizontal=True,
        key="sel_voice_manual",
    )
    manual_voice_code = constants.VOICE_MAP[manual_voice_label]

    enable_audio = st.checkbox("启用语音", value=True, key="chk_audio_manual")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_gen = st.button("🚀 生成卡片", type="primary", use_container_width=True)
    with col_btn2:
        st.button("🗑️ 清空重置", type="secondary", on_click=reset_anki_state, key="btn_clear_anki")

    if start_gen:
        if not ai_response.strip():
            st.warning("⚠️ 输入框为空。")
        else:
            progress_container = st.container()
            with progress_container:
                progress_bar_manual = st.progress(0)
                status_manual = st.empty()

            def update_progress_manual(ratio: float, text: str) -> None:
                progress_bar_manual.progress(ratio)
                status_manual.text(text)

            with st.spinner("⏳ 正在解析并生成..."):
                parsed_data = parse_anki_data(ai_response)
                if parsed_data:
                    st.session_state['anki_cards_cache'] = parsed_data
                    try:
                        file_path = generate_anki_package(
                            parsed_data,
                            deck_name,
                            enable_tts=enable_audio,
                            tts_voice=manual_voice_code,
                            progress_callback=update_progress_manual
                        )

                        set_anki_pkg(file_path, deck_name)

                        status_manual.markdown(f"✅ **生成完毕！共制作 {len(parsed_data)} 张卡片**")
                        st.balloons()
                        st.toast("任务完成！", icon="🎉")
                        run_gc()
                    except Exception as e:
                        from errors import ErrorHandler
                        ErrorHandler.handle(e, "生成文件出错")
                else:
                    st.error("❌ 解析失败，请检查输入格式。")

    if st.session_state['anki_cards_cache']:
        cards = st.session_state['anki_cards_cache']
        with st.expander(f"👀 预览卡片 (前 {constants.MAX_PREVIEW_CARDS} 张)", expanded=False):
            df_view = pd.DataFrame(cards)
            display_cols = ['w', 'm', 'e', 'r']
            df_view = df_view[[c for c in display_cols if c in df_view.columns]]
            col_labels = ["正面", "中文/英文释义", "例句"]
            if len(df_view.columns) > 3:
                col_labels.append("词源")
            df_view.columns = col_labels[:len(df_view.columns)]
            st.dataframe(df_view.head(constants.MAX_PREVIEW_CARDS), use_container_width=True, hide_index=True)

        render_anki_download_button(
            f"📥 下载 {st.session_state.get('anki_pkg_name', 'deck.apkg')}",
            button_type="primary"
        )

st.markdown(
    '<div class="app-footer">Vocab Flow Ultra &nbsp;·&nbsp; Built for learners</div>',
    unsafe_allow_html=True
)
