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
from errors import ErrorHandler
from extraction import (
    extract_text_from_file,
    extract_text_from_url,
    is_upload_too_large,
    parse_anki_txt_export,
)
from state import set_generated_words_state
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
    page_title="Vocab Flow Ultra · 词汇助手",
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
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-size-adjust: 100%;
        -webkit-text-size-adjust: 100%;
        font-size: 17px;
    }

    /* iOS safe-area support */
    .main .block-container {
        padding-left: max(1rem, env(safe-area-inset-left));
        padding-right: max(1rem, env(safe-area-inset-right));
        padding-bottom: max(1rem, env(safe-area-inset-bottom));
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
        min-height: 44px;
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }
    .stButton>button:active { transform: translateY(0); }

    /* ===== Text areas ===== */
    .stTextArea textarea {
        font-family: 'Consolas', 'SF Mono', 'Monaco', monospace;
        font-size: 16px; border-radius: 10px;
        -webkit-overflow-scrolling: touch;
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
        font-weight: 700; letter-spacing: -0.02em; font-size: 1.45rem;
    }

    /* ===== Tabs: segmented-control style ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #f1f5f9; padding: 4px;
        border-radius: 12px; border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 1rem; border-radius: 10px;
        font-weight: 600; font-size: 1rem;
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
        font-size: 2.45rem; font-weight: 900; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.2rem;
    }
    .app-hero p {
        color: #64748b; font-size: 1.02rem; margin: 0;
        font-weight: 600;
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
    .stTextInput input, .stSelectbox input, .stNumberInput input {
        min-height: 42px;
        font-size: 16px;
    }

    /* ===== Toast / info / warning boxes ===== */
    .stAlert { border-radius: 10px; }
    .stMarkdown p, .stCaption, label, .stRadio label, .stCheckbox label {
        font-size: 1rem;
    }

    /* ===== Download button ===== */
    .stDownloadButton > button {
        border-radius: 10px; font-weight: 600;
        box-shadow: 0 2px 8px rgba(59,130,246,0.15);
        min-height: 44px;
    }

    /* ===== Dark mode refinement ===== */
    @media (prefers-color-scheme: dark) {
        .stApp { background: #0b1220; color: #e5e7eb; }
        .stForm {
            background: #111827;
            border-color: #1f2937;
            box-shadow: 0 1px 4px rgba(0,0,0,0.35);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
            border-color: #1f2937;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: #0f172a;
            border-color: #1f2937;
        }
        .stTabs [data-baseweb="tab"] {
            color: #cbd5e1;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #111827;
            color: #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.35);
        }
        .stExpander {
            border-color: #1f2937;
            box-shadow: 0 1px 2px rgba(0,0,0,0.35);
        }
        .stAlert {
            background: #0f172a;
            border-color: #1f2937;
            color: #e5e7eb;
        }
        hr { background: #1f2937; }
        .app-footer {
            color: #94a3b8;
            border-top-color: #1f2937;
        }
        .app-hero p { color: #94a3b8; }
        .stRadio > div > label {
            border-color: #334155;
            color: #e5e7eb;
            background: #0f172a;
        }
        .stRadio > div > label:hover {
            border-color: #3b82f6;
            background: #172554;
        }
    }

    /* ===== Mobile (iOS/Android) ===== */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.8rem;
            padding-left: max(0.75rem, env(safe-area-inset-left));
            padding-right: max(0.75rem, env(safe-area-inset-right));
            padding-bottom: max(1rem, env(safe-area-inset-bottom));
        }
        .app-hero {
            padding: 0.9rem 0 0.2rem;
        }
        .app-hero h1 {
            font-size: 1.95rem;
        }
        .app-hero p {
            font-size: 0.92rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.55rem 0.75rem;
            font-size: 0.9rem;
            min-height: 40px;
        }
        .stButton>button,
        .stDownloadButton > button {
            min-height: 46px;
            border-radius: 12px;
        }
        /* Avoid hover lift on touch devices */
        .stButton>button:hover {
            transform: none;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .stCaption {
            font-size: 0.9rem;
        }
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


def _rank_badge_style(rank: int) -> tuple[str, str]:
    """Map numeric rank to badge color and label."""
    if rank <= 5000:
        return "#10b981", "高频词"
    if rank <= 10000:
        return "#3b82f6", "常用词"
    if rank <= 20000:
        return "#f59e0b", "进阶词"
    if rank < 99999:
        return "#ef4444", "专业词"
    return "#6b7280", "未收录"


def _analyze_and_set_words(raw_text: str, min_rank: int, max_rank: int) -> bool:
    """Run rank-based analysis and update session state. Returns success."""
    if len(raw_text) <= 2:
        return False
    final_data, raw_count, stats_info = analyze_logic(raw_text, min_rank, max_rank, False)
    set_generated_words_state(final_data, raw_count, stats_info)
    return True


st.markdown("""
<div class="app-hero">
    <h1>词汇助手</h1>
    <p>查词、筛词、制卡一体化</p>
</div>
""", unsafe_allow_html=True)


def _do_lookup(query_word: str) -> None:
    """Execute AI lookup for a word, populating session state cache and result."""
    st.session_state["quick_lookup_is_loading"] = True
    try:
        cache_key = f"lookup_cache_{query_word.lower()}"
        if cache_key not in st.session_state:
            stream_box = st.empty()

            def _on_stream(text: str) -> None:
                safe = html.escape(text).replace("\n", "<br>")
                stream_box.markdown(
                    (
                        '<div style="padding:10px 12px;border:1px solid #dbeafe;'
                        'background:#f8fbff;border-radius:10px;color:#1f2937;'
                        'line-height:1.7;font-size:16px;">'
                        f'{safe}'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

            with st.spinner("🔍 查询中..."):
                st.session_state[cache_key] = get_word_quick_definition(
                    query_word,
                    stream_callback=_on_stream,
                )
            stream_box.empty()
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

    pending_word = st.session_state.pop("_quick_lookup_pending_word", "")
    if pending_word:
        st.session_state["quick_lookup_word"] = pending_word
        st.session_state["_auto_lookup_word"] = pending_word

    auto_word = st.session_state.pop("_auto_lookup_word", "")
    if auto_word and not in_cooldown:
        _do_lookup(auto_word)

    _btn_label = "查询中..." if st.session_state["quick_lookup_is_loading"] else "🔍 deepseek"

    with st.form("quick_lookup_form", clear_on_submit=False, border=False):
        col_word, col_btn = st.columns([4, 1])
        with col_word:
            lookup_word = st.text_input(
                "输入单词或短语",
                placeholder="输入单词或短语，回车查询 …",
                key="quick_lookup_word",
                label_visibility="collapsed",
                autocomplete="off",
            )
        with col_btn:
            lookup_submit = st.form_submit_button(
                _btn_label,
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

        rank_color, rank_label = _rank_badge_style(rank)

        # Build styled HTML lines (no iframe needed)
        lines = raw_content.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

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

    elif result and 'error' in result:
        # Avoid flashing red error blocks on reruns/refresh.
        st.toast(f"查词失败：{result.get('error', '未知错误')}", icon="⚠️")
        st.session_state["quick_lookup_last_result"] = None

    st.markdown("---")


if hasattr(st, "fragment"):
    render_quick_lookup = st.fragment(render_quick_lookup)

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv` 词库文件，请检查目录。")

with st.expander("使用指南 & 支持格式", expanded=False):
    st.markdown("""
    **极速工作流**
    1. **查词** — 顶部 AI 查词，秒速获取精准释义、词源拆解和双语例句
    2. **提取** — 支持 PDF / ePub / Docx / TXT / CSV / Excel 等格式
    3. **生成** — AI 释义 + 并发语音合成，一键打包下载

    **支持的文件格式**
    TXT · PDF · DOCX · EPUB · CSV · XLSX · XLS · DB · SQLite · Anki 导出 (.txt)
    """)

tab_lookup, tab_extract, tab_anki = st.tabs([
    "AI查词",
    "筛选单词",
    "anki制卡",
])

with tab_lookup:
    render_quick_lookup()

# ==========================================
# Tab 1: Word Extraction
# ==========================================
with tab_extract:
    mode_paste, mode_url, mode_upload, mode_rank, mode_manual = st.tabs([
        "文本",
        "链接",
        "文件",
        "词库",
        "词表",
    ])

    with mode_paste:
        col1, col2 = st.columns(2)
        current_rank = col1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100)
        target_rank = col2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500)

        if target_rank < current_rank:
            st.warning("⚠️ Max Rank 必须大于等于 Min Rank")

        pasted_text = st.text_area(
            "粘贴文章文本",
            height=100,
            key="paste_key_2_1",
            placeholder="支持直接粘贴文章内容..."
        )

        if st.button("🚀 从文本生成重点词", type="primary", key="btn_mode_2_1"):
            if target_rank < current_rank:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请修正后重试。")
            else:
                with st.status("🔍 正在加载资源并分析文本...", expanded=True) as status:
                    start_time = time.time()
                    raw_text = pasted_text

                    status.write("🧠 正在进行 NLP 词形还原与分级...")
                    if _analyze_and_set_words(raw_text, current_rank, target_rank):
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        status.update(label="✅ 分析完成", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 内容为空或太短", state="error")

    with mode_url:
        col1, col2 = st.columns(2)
        current_rank_url = col1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100, key="min_rank_2_2")
        target_rank_url = col2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500, key="max_rank_2_2")

        if target_rank_url < current_rank_url:
            st.warning("⚠️ Max Rank 必须大于等于 Min Rank")

        input_url = st.text_input(
            "🔗 输入文章 URL（自动抓取）",
            placeholder="https://www.economist.com/...",
            key="url_input_key_2_2"
        )

        if st.button("🌐 从链接生成重点词", type="primary", key="btn_mode_2_2"):
            if target_rank_url < current_rank_url:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请修正后重试。")
            elif not input_url.strip():
                st.warning("⚠️ 请输入有效链接。")
            else:
                with st.status("🌐 正在抓取并分析网页文本...", expanded=True) as status:
                    start_time = time.time()
                    status.write(f"正在抓取：{input_url}")
                    raw_text = extract_text_from_url(input_url)
                    if _analyze_and_set_words(raw_text, current_rank_url, target_rank_url):
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        status.update(label="✅ 生成完成", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 抓取内容为空或过短", state="error")

    with mode_upload:
        col1, col2 = st.columns(2)
        current_rank_upload = col1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100, key="min_rank_2_3")
        target_rank_upload = col2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500, key="max_rank_2_3")

        if target_rank_upload < current_rank_upload:
            st.warning("⚠️ Max Rank 必须大于等于 Min Rank")

        uploaded_file = st.file_uploader(
            "上传文件（TXT/PDF/DOCX/EPUB/CSV/Excel/DB）",
            type=['txt', 'pdf', 'docx', 'epub', 'csv', 'xlsx', 'xls', 'db', 'sqlite'],
            key="upload_2_3",
        )
        if uploaded_file and is_upload_too_large(uploaded_file):
            st.error(f"❌ 文件过大，已限制为 {constants.MAX_UPLOAD_MB}MB。请缩小文件后重试。")
            uploaded_file = None

        if st.button("📁 从文件生成重点词", type="primary", key="btn_mode_2_3"):
            if target_rank_upload < current_rank_upload:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请修正后重试。")
            elif uploaded_file is None:
                st.warning("⚠️ 请先上传文件。")
            else:
                with st.status("📄 正在解析文件并提取重点词...", expanded=True) as status:
                    start_time = time.time()
                    raw_text = extract_text_from_file(uploaded_file)
                    if _analyze_and_set_words(raw_text, current_rank_upload, target_rank_upload):
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        status.update(label="✅ 生成完成", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 文件内容为空或过短", state="error")

    with mode_manual:
        st.markdown("#### 直接粘贴整理好的词表（不做 rank 筛选）")
        manual_words_text = st.text_area(
            "✍️ 单词列表（每行一个或逗号分隔）",
            height=220,
            key="manual_words_2_5",
            placeholder="altruism\nhectic\nserendipity",
        )

        if st.button("🧾 生成词表（不筛 rank）", key="btn_mode_2_5", type="primary"):
            with st.spinner("正在解析列表..."):
                if manual_words_text.strip():
                    words = [w.strip() for w in re.split(r'[,\n\t]+', manual_words_text) if w.strip()]
                    unique_words = []
                    seen = set()

                    for word in words:
                        w_lower = word.lower().strip()
                        if not w_lower or w_lower in seen:
                            continue
                        seen.add(w_lower)
                        unique_words.append(word)

                    raw_count = len(words)
                    data_list = [(w, VOCAB_DICT.get(w.lower(), 99999)) for w in unique_words]
                    set_generated_words_state(data_list, raw_count, None)
                    duplicated = raw_count - len(unique_words)
                    msg = f"✅ 已加载 {len(unique_words)} 个单词（不筛 rank）"
                    if duplicated > 0:
                        msg += f"（去重 {duplicated} 个）"
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
                card_text = st.empty()
                card_bar = st.progress(0.0)
                audio_text = st.empty()
                audio_bar = st.progress(0.0)

                progress_title.markdown("#### ⏳ 内置 AI 制卡进度")
                card_text.markdown("**制卡进度**：AI 生成中...")
                audio_text.markdown("**音频进度**：等待制卡完成...")

                def update_ai_progress(current: int, total: int) -> None:
                    ratio = current / total if total > 0 else 0.0
                    card_bar.progress(min(0.9, ratio * 0.9))
                    card_text.markdown(f"**制卡进度**：AI 生成中（{current}/{total}）")

                ai_result = process_ai_in_batches(
                    words_for_auto_ai,
                    progress_callback=update_ai_progress,
                )

                if ai_result:
                    card_text.markdown("**制卡进度**：正在解析 AI 结果...")
                    parsed_data = parse_anki_data(ai_result)

                    if parsed_data:
                        try:
                            card_bar.progress(1.0)
                            card_text.markdown(f"**制卡进度**：✅ 完成（共 {len(parsed_data)} 张）")
                            audio_text.markdown("**音频进度**：进行中...")
                            audio_bar.progress(0.0)
                            deck_name = f"Vocab_{get_beijing_time_str()}"

                            def update_pkg_progress(ratio: float, text: str) -> None:
                                audio_bar.progress(ratio)
                                audio_text.markdown(f"**音频进度**：{text}")

                            file_path = generate_anki_package(
                                parsed_data,
                                deck_name,
                                enable_tts=enable_audio_auto,
                                tts_voice=selected_voice_code,
                                progress_callback=update_pkg_progress
                            )

                            set_anki_pkg(file_path, deck_name)

                            audio_bar.progress(1.0)
                            audio_text.markdown("**音频进度**：✅ 完成")
                            st.balloons()
                            run_gc()
                        except Exception as e:
                            audio_text.markdown("**音频进度**：❌ 失败")
                            ErrorHandler.handle(e, "生成出错")
                    else:
                        card_text.markdown("**制卡进度**：❌ 解析失败")
                        audio_text.markdown("**音频进度**：未开始")
                        st.error("解析失败，AI 返回内容为空或格式错误。")
                else:
                    card_text.markdown("**制卡进度**：❌ AI 生成失败")
                    audio_text.markdown("**音频进度**：未开始")
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
                st.markdown("#### ⚙️ 卡片格式自定义")
                col_front, col_def = st.columns(2)
                with col_front:
                    front_val = st.radio(
                        "正面内容",
                        options=["word", "phrase"],
                        format_func=lambda v: "📝 单词" if v == "word" else "📝 短语/搭配",
                        index=1,
                        horizontal=True,
                        key="tab1_prompt_front_v2",
                    )
                with col_def:
                    def_val = st.radio(
                        "释义语言",
                        options=["cn", "en", "both"],
                        format_func=lambda v: {
                            "cn": "🇨🇳 中文释义",
                            "en": "🇬🇧 英文释义",
                            "both": "🇨🇳🇬🇧 中英双语",
                        }[v],
                        index=0,
                        horizontal=True,
                        key="tab1_prompt_def_v2",
                    )

                col_ex, col_ety = st.columns(2)
                with col_ex:
                    ex_val = st.radio(
                        "例句数量",
                        options=[1, 2, 3],
                        format_func=lambda v: f"{v} 个例句",
                        index=0,
                        horizontal=True,
                        key="tab1_prompt_ex_v2",
                    )
                with col_ety:
                    ety_val = st.radio(
                        "词源词根",
                        options=[True, False],
                        format_func=lambda v: "✅ 包含词源" if v else "❌ 不含词源",
                        index=0,
                        horizontal=True,
                        key="tab1_prompt_ety_v2",
                    )

                card_fmt: CardFormat = {
                    "front": front_val,
                    "definition": def_val,
                    "examples": ex_val,
                    "etymology": ety_val,
                }
                st.caption(
                    f"当前格式：正面={front_val} ｜ 释义={def_val} ｜ 例句={ex_val} ｜ 词源={'on' if ety_val else 'off'}"
                )
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
