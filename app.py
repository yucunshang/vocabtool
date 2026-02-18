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
)
from rate_limiter import (
    check_batch_limit, check_lookup_limit, check_url_limit,
    record_batch, record_lookup, record_url,
)
from state import set_generated_words_state
from ui_styles import APP_STYLES_HTML
from utils import get_beijing_time_str, render_copy_button, run_gc
from vocab import analyze_logic

# Load vocab data for use in mode_rank tab and the vocab error check below.
VOCAB_DICT, FULL_DF = resources.load_vocab_data()

# Clean old .apkg files from our temp subdir (e.g. from previous sessions)
cleanup_old_apkg_files()

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

# Logging: ensure root logger has a handler when running as main app
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
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

# Custom CSS – app-like design (see ui_styles.py)
st.markdown(APP_STYLES_HTML, unsafe_allow_html=True)


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
        st.caption("💡 如下载无反应，请在浏览器（Safari / Chrome）中打开本页面再点击下载。")
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
    if not final_data:
        st.info(
            f"📭 共提取 {raw_count} 个原始词，经 rank {min_rank}–{max_rank} 筛选后无剩余单词。"
            " 可尝试扩大 rank 范围，或使用「词表」模式直接粘贴。"
        )
        return False
    return True


st.markdown("""
<div class="app-hero">
    <h1>词汇助手</h1>
    <p>查词、筛词、制卡一体化</p>
</div>
""", unsafe_allow_html=True)


def _render_extraction_stats(data: list) -> None:
    """Show 4 metrics: coverage, target density, raw count, filtered count."""
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


def _render_word_editor(data: list) -> list:
    """Render editable word list textarea with copy button. Returns current words_only."""
    words_only = [w for w, r in data]
    words_text = "\n".join(words_only)
    original_count = len(data)
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
        return edited_word_list
    else:
        return words_only


def _render_audio_settings(key_prefix: str) -> tuple[bool, str]:
    """Render audio toggle + voice selector. Returns (enable_audio, voice_code)."""
    enable_audio = st.checkbox("启用语音", value=True, key=f"chk_audio_{key_prefix}")
    if enable_audio:
        selected_voice_label = st.radio(
            "🎙️ 发音人",
            options=list(constants.VOICE_MAP.keys()),
            index=0,
            horizontal=True,
            key=f"sel_voice_{key_prefix}",
        )
        voice_code = constants.VOICE_MAP[selected_voice_label]
    else:
        voice_code = list(constants.VOICE_MAP.values())[0]
    return enable_audio, voice_code


def _render_builtin_ai_section(
    words_only: list, enable_audio: bool, voice_code: str,
    card_format, use_builtin_ai: str
) -> None:
    """Builtin AI generation flow: button → progress → parse → package → download."""
    ai_model_label = get_config()["openai_model"]

    words_for_auto_ai = words_only
    current_word_count = len(words_for_auto_ai)
    if current_word_count > constants.MAX_AUTO_LIMIT:
        st.caption(
            f"⚠️ 当前 {current_word_count} 词；内置 AI 最多处理前 {constants.MAX_AUTO_LIMIT} 词。"
            " 可选「第三方 AI」复制 Prompt 分批处理。"
        )
        words_for_auto_ai = words_for_auto_ai[:constants.MAX_AUTO_LIMIT]

    if use_builtin_ai == "builtin":
        if st.button(f"🚀 使用 {ai_model_label} 生成", type="primary", use_container_width=True):
            batch_allowed, batch_msg = check_batch_limit()
            if not batch_allowed:
                st.warning(batch_msg)
                st.stop()
            record_batch()

            progress_title = st.empty()
            card_text = st.empty()
            card_bar = st.progress(0.0)
            audio_text = st.empty()
            audio_bar = st.progress(0.0)

            total_words = len(words_for_auto_ai)
            batch_size = constants.AI_BATCH_SIZE
            first_end = min(batch_size, total_words)
            progress_title.markdown("#### ⏳ 内置 AI 制卡进度")
            card_text.markdown(f"**制卡进度**：第 1 组（1–{first_end}/{total_words}）AI 生成中...")
            audio_text.markdown("**音频进度**：等待制卡完成...")

            def update_ai_progress(current: int, total: int) -> None:
                ratio = current / total if total > 0 else 0.0
                card_bar.progress(min(0.9, ratio * 0.9))
                batch_idx = (current + batch_size - 1) // batch_size
                start = (batch_idx - 1) * batch_size + 1
                end = min(batch_idx * batch_size, total)
                card_text.markdown(f"**制卡进度**：第 {batch_idx} 组（{start}–{end}/{total}）AI 生成中...")

            ai_result = process_ai_in_batches(
                words_for_auto_ai,
                progress_callback=update_ai_progress,
                card_format=card_format,
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
                            enable_tts=enable_audio,
                            tts_voice=voice_code,
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
    else:
        st.info("请使用右侧「复制 Prompt」到第三方 AI，格式与上方通用设置一致。")

    render_anki_download_button(
        f"📥 下载 {st.session_state.get('anki_pkg_name', 'deck.apkg')}",
        button_type="primary",
        use_container_width=True
    )
    st.caption("⚠️ AI 结果请人工复核后再学习。")


def _render_thirdparty_prompt_section(
    words_only: list, examples_colloquial: bool, use_builtin_ai: str
) -> None:
    """Render card-format options and a copyable prompt block for third-party AI services."""
    if use_builtin_ai == "thirdparty":
        st.markdown("#### 📌 复制 Prompt（可自定义卡片格式）")
        st.caption("在下方选择卡片格式后复制 Prompt 到第三方 AI，生成结果粘贴到「Anki 制卡」页解析。")
    else:
        st.markdown("#### 第三方 AI Prompt")
        st.caption("需要大批量或自定义卡片格式时，可切换为「第三方 AI」在下方自定义格式并复制 Prompt。")

    with st.expander("📌 复制 Prompt（第三方 AI）", expanded=(use_builtin_ai == "thirdparty")):
        st.markdown("##### ⚙️ 卡片格式（仅影响下方 Prompt）")
        col_tp_front, col_tp_def = st.columns(2)
        with col_tp_front:
            tp_front = st.radio(
                "正面",
                options=["word", "phrase"],
                format_func=lambda x: "单词" if x == "word" else "短语/词组",
                index=0,
                horizontal=True,
                key="tp_prompt_front",
            )
        with col_tp_def:
            tp_def = st.radio(
                "释义",
                options=["cn", "en", "both"],
                format_func=lambda x: {"cn": "中文", "en": "英文", "both": "中英双语"}[x],
                index=0,
                horizontal=True,
                key="tp_prompt_def",
            )
        col_tp_ex, col_tp_ety = st.columns(2)
        with col_tp_ex:
            tp_ex = st.radio(
                "例句数量",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} 条",
                index=1,
                horizontal=True,
                key="tp_prompt_ex",
            )
        with col_tp_ety:
            tp_ety = st.checkbox("词根词源词缀", value=False, key="tp_prompt_ety")
        tp_ex_cn = st.checkbox("例句带中文翻译", value=True, key="tp_prompt_ex_cn")
        tp_colloquial = st.checkbox("例句用口语", value=examples_colloquial, key="tp_prompt_colloquial", help="例句使用日常口语化表达")

        third_party_card_format: CardFormat = {
            "front": tp_front,
            "definition": tp_def,
            "examples": tp_ex,
            "examples_with_cn": tp_ex_cn,
            "etymology": tp_ety,
            "examples_colloquial": tp_colloquial,
        }

        batch_size_prompt = int(
            st.number_input("🔢 分组大小 (最大 500)", min_value=1, max_value=500, value=50, step=10, key="batch_size_prompt")
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
        strict_prompt_template = build_card_prompt(words_str_for_prompt, third_party_card_format)
        st.code(strict_prompt_template, language="text")


def _render_extract_results() -> None:
    """Render the extracted words results, AI card generation, and third-party prompt."""
    if not st.session_state.get('gen_words_data'):
        return

    data = st.session_state['gen_words_data']

    _render_extraction_stats(data)

    st.markdown("### ✅ 提取成功！")

    words_only = _render_word_editor(data)

    st.markdown("---")
    st.markdown("### 🤖 AI 生成 Anki 卡片")

    # ---------- 公用设置（内置 AI 与第三方 Prompt 共用）----------
    st.markdown("#### ① 通用设置")
    enable_audio, voice_code = _render_audio_settings("auto")

    examples_colloquial = st.checkbox(
        "例句用口语",
        value=False,
        key="chk_examples_colloquial",
        help="例句使用日常口语化表达，而非书面语",
    )

    # 固定模板：正面单词，反面中文释义 + 2 条例句带中文翻译，不加词根词缀
    shared_card_format: CardFormat = {
        "front": "word",
        "definition": "cn",
        "examples": 2,
        "examples_with_cn": True,
        "etymology": False,
        "examples_colloquial": examples_colloquial,
    }

    st.markdown("#### ② 生成方式")
    use_builtin_ai = st.radio(
        "选择",
        options=["builtin", "thirdparty"],
        format_func=lambda x: "内置 AI 一键生成" if x == "builtin" else "第三方 AI（复制 Prompt）",
        index=0,
        horizontal=True,
        key="ai_gen_mode",
    )

    col_ai_btn, col_copy_hint = st.columns([1, 1.35], vertical_alignment="top")

    with col_ai_btn:
        _render_builtin_ai_section(words_only, enable_audio, voice_code, shared_card_format, use_builtin_ai)

    with col_copy_hint:
        _render_thirdparty_prompt_section(words_only, examples_colloquial, use_builtin_ai)


def _do_lookup(query_word: str) -> None:
    """Execute AI lookup for a word, populating session state cache and result."""
    # Input length guard
    if len(query_word) > constants.MAX_LOOKUP_INPUT_LENGTH:
        st.warning(f"⚠️ 输入过长（最多 {constants.MAX_LOOKUP_INPUT_LENGTH} 字符）。")
        return

    # Rate limit check
    allowed, msg = check_lookup_limit()
    if not allowed:
        st.warning(msg)
        return

    st.session_state["quick_lookup_is_loading"] = True
    try:
        # Case-sensitive cache so "China" vs "china", "May" vs "may" get different results
        cache_key = f"lookup_cache_{query_word.strip()}"
        if cache_key not in st.session_state:
            stream_box = st.empty()

            def _on_stream(text: str) -> None:
                safe = html.escape(text).replace("\n", "<br>")
                stream_box.markdown(
                    f'<div class="ql-stream">{safe}</div>',
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
        record_lookup()
    finally:
        st.session_state["quick_lookup_is_loading"] = False
        st.session_state["quick_lookup_block_until"] = time.time() + constants.QUICK_LOOKUP_COOLDOWN_SECONDS


def render_quick_lookup() -> None:
    # Guard against sessions that predate DEFAULT_SESSION_STATE centralization.
    for _k, _v in {
        "quick_lookup_last_query": "",
        "quick_lookup_last_result": None,
        "quick_lookup_is_loading": False,
        "quick_lookup_block_until": 0.0,
        "quick_lookup_cache_keys": [],
    }.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    now_ts = time.time()
    in_cooldown = now_ts < st.session_state["quick_lookup_block_until"]
    lookup_disabled = st.session_state["quick_lookup_is_loading"] or in_cooldown

    pending_word = st.session_state.pop("_quick_lookup_pending_word", "")
    if pending_word:
        st.session_state["quick_lookup_word"] = pending_word
        st.session_state["_auto_lookup_word"] = pending_word

    # Handle pending clear (must happen before text_input widget is created)
    if st.session_state.pop("_quick_lookup_pending_clear", False):
        st.session_state["quick_lookup_word"] = ""
        st.session_state["quick_lookup_last_query"] = ""
        st.session_state["quick_lookup_last_result"] = None
        st.toast("已清空", icon="🗑️")

    auto_word = st.session_state.pop("_auto_lookup_word", "")
    if auto_word and not in_cooldown:
        _do_lookup(auto_word)

    _btn_label = "查询中..." if st.session_state["quick_lookup_is_loading"] else "🔍 deepseek"
    _has_content = bool(st.session_state.get("quick_lookup_word") or st.session_state.get("quick_lookup_last_result"))

    with st.form("quick_lookup_form", clear_on_submit=False, border=False):
        if _has_content:
            col_word, col_btn, col_clear = st.columns([4, 2, 1.2])
        else:
            col_word, col_btn = st.columns([5, 2])
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
        if _has_content:
            with col_clear:
                clear_submit = st.form_submit_button("清空", use_container_width=True)
        else:
            clear_submit = False
        if clear_submit:
            st.session_state["_quick_lookup_pending_clear"] = True
            st.rerun()

    if in_cooldown:
        wait_seconds = max(0.0, st.session_state["quick_lookup_block_until"] - now_ts)
        st.caption(f"⏱️ 请稍候 {wait_seconds:.1f}s 再次查询")

    if lookup_submit and not clear_submit:
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
            safe = html.escape(line)

            if line.startswith("🌱"):
                formatted_lines.append(f'<div class="ql-etym">{safe}</div>')
            elif "|" in line and len(line) < 50:
                formatted_lines.append(f'<div class="ql-def">{safe}</div>')
            elif line.startswith("•"):
                formatted_lines.append(f'<div class="ql-ex">{safe}</div>')
            else:
                formatted_lines.append(f'<div class="ql-misc">{safe}</div>')

        display_html = "".join(formatted_lines)
        rank_badge = f'<span style="display:inline-block;background:{rank_color};color:white;padding:3px 10px;border-radius:5px;font-size:13px;font-weight:600;">📊 {rank} · {rank_label}</span>'

        st.markdown(f'<div class="ql-result">{display_html}<div style="margin-top:10px;">{rank_badge}</div></div>', unsafe_allow_html=True)

    elif result and 'error' in result:
        # Avoid flashing red error blocks on reruns/refresh.
        st.toast(f"查词失败：{result.get('error', '未知错误')}", icon="⚠️")
        st.session_state["quick_lookup_last_result"] = None

    st.markdown("---")


if hasattr(st, "fragment"):
    render_quick_lookup = st.fragment(render_quick_lookup)
    _render_extract_results = st.fragment(_render_extract_results)

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
def _render_rank_inputs(key_suffix: str) -> tuple[int, int]:
    """Render min/max rank number inputs with validation warning. Returns (min_rank, max_rank)."""
    col1, col2 = st.columns(2)
    min_rank = col1.number_input(
        "忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100,
        key=f"min_rank_{key_suffix}"
    )
    max_rank = col2.number_input(
        "忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500,
        key=f"max_rank_{key_suffix}"
    )
    if max_rank < min_rank:
        st.warning("⚠️ Max Rank 必须大于等于 Min Rank")
    return min_rank, max_rank


with tab_extract:
    mode_paste, mode_url, mode_upload, mode_rank, mode_manual = st.tabs([
        "文本",
        "链接",
        "文件",
        "词库",
        "词表",
    ])

    with mode_paste:
        current_rank, target_rank = _render_rank_inputs("2_1")

        pasted_text = st.text_area(
            "粘贴文章文本",
            height=100,
            key="paste_key_2_1",
            placeholder="支持直接粘贴文章内容..."
        )

        col_gen_p, col_clr_p = st.columns([4, 1])
        with col_gen_p:
            btn_paste = st.button("🚀 从文本生成重点词", type="primary", key="btn_mode_2_1", use_container_width=True)
        with col_clr_p:
            st.button("清空", key="clr_paste", use_container_width=True,
                       on_click=lambda: st.session_state.update({"paste_key_2_1": ""}))

        if btn_paste:
            if target_rank < current_rank:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请修正后重试。")
            elif len(pasted_text) > constants.MAX_PASTE_TEXT_LENGTH:
                st.error(f"❌ 文本过长（最大约 {constants.MAX_PASTE_TEXT_LENGTH // 1000}K 字符），请缩短后重试。")
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
        current_rank_url, target_rank_url = _render_rank_inputs("2_2")

        input_url = st.text_input(
            "🔗 输入文章 URL（自动抓取）",
            placeholder="https://www.economist.com/...",
            key="url_input_key_2_2"
        )

        col_gen_u, col_clr_u = st.columns([4, 1])
        with col_gen_u:
            btn_url = st.button("🌐 从链接生成重点词", type="primary", key="btn_mode_2_2", use_container_width=True)
        with col_clr_u:
            st.button("清空", key="clr_url", use_container_width=True,
                       on_click=lambda: st.session_state.update({"url_input_key_2_2": ""}))

        if btn_url:
            if target_rank_url < current_rank_url:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请修正后重试。")
            elif not input_url.strip():
                st.warning("⚠️ 请输入有效链接。")
            elif len(input_url) > constants.MAX_URL_LENGTH:
                st.error(f"❌ URL 过长（最大 {constants.MAX_URL_LENGTH} 字符）。")
            elif not re.match(r'^https?://', input_url.strip()):
                st.error("❌ 请输入以 http:// 或 https:// 开头的有效链接。")
            else:
                url_allowed, url_msg = check_url_limit()
                if not url_allowed:
                    st.warning(url_msg)
                else:
                    record_url()
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
        current_rank_upload, target_rank_upload = _render_rank_inputs("2_3")

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
                    truncated = False
                    if len(raw_text) > constants.MAX_TEXT_ANALYSIS_CHARS:
                        raw_text = raw_text[: constants.MAX_TEXT_ANALYSIS_CHARS]
                        truncated = True
                    if _analyze_and_set_words(raw_text, current_rank_upload, target_rank_upload):
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        label = "✅ 生成完成（已仅分析前 30 万字）" if truncated else "✅ 生成完成"
                        status.update(label=label, state="complete", expanded=False)
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

        col_gen_m, col_clr_m = st.columns([4, 1])
        with col_gen_m:
            btn_gen_manual = st.button("🧾 生成词表（不筛 rank）", key="btn_mode_2_5", type="primary", use_container_width=True)
        with col_clr_m:
            st.button("清空", key="clr_manual_words", use_container_width=True,
                       on_click=lambda: st.session_state.update({"manual_words_2_5": ""}))

        if btn_gen_manual:
            with st.spinner("正在解析列表..."):
                if manual_words_text.strip():
                    raw_items = [w.strip() for w in re.split(r'[,\n\t]+', manual_words_text) if w.strip()]
                    valid_words = []
                    invalid_items = []
                    for item in raw_items:
                        if re.match(r"^[a-zA-Z]+(?:[-'][a-zA-Z]+)*$", item):
                            valid_words.append(item)
                        else:
                            invalid_items.append(item)

                    if invalid_items:
                        preview = ", ".join(invalid_items[:10])
                        suffix = f" 等共 {len(invalid_items)} 项" if len(invalid_items) > 10 else ""
                        st.warning(f"⚠️ 已跳过格式不正确的条目：{preview}{suffix}")

                    if not valid_words:
                        st.error("❌ 没有识别到有效的英文单词，请检查输入格式（每行一个单词或逗号分隔）。")
                    else:
                        unique_words = []
                        seen = set()
                        for word in valid_words:
                            w_lower = word.lower().strip()
                            if not w_lower or w_lower in seen:
                                continue
                            seen.add(w_lower)
                            unique_words.append(word)

                        raw_count = len(valid_words)
                        data_list = [(w, resources.get_rank_for_word(w)) for w in unique_words]
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
                        rank_col = next((c for c in FULL_DF.columns if 'rank' in c), None)
                        word_col = next((c for c in FULL_DF.columns if 'word' in c), None)
                        if rank_col is None or word_col is None:
                            st.error("❌ 词库CSV格式异常：缺少 rank 或 word 列")
                        else:
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
                            rank_col = next((c for c in FULL_DF.columns if 'rank' in c), None)
                            word_col = next((c for c in FULL_DF.columns if 'word' in c), None)
                            if rank_col is None or word_col is None:
                                st.error("❌ 词库CSV格式异常：缺少 rank 或 word 列")
                            else:
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
    _render_extract_results()

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
