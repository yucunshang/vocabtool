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
from state import export_session_json, import_session_json, set_generated_words_state
from ui_styles import APP_STYLES_HTML
from utils import get_beijing_time_str, render_copy_button, render_prompt_copy_button, run_gc
from vocab import analyze_logic

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

# Load vocab data for use in mode_rank tab and the vocab error check below.
VOCAB_DICT, FULL_DF = resources.load_vocab_data()

# Clean old .apkg files from our temp subdir (e.g. from previous sessions)
cleanup_old_apkg_files()

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


@st.cache_data(show_spinner=False)
def _cached_analyze(text: str, min_rank: int, max_rank: int, include_unknown: bool):
    return analyze_logic(text, min_rank, max_rank, include_unknown)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_extract_url(url: str) -> str:
    return extract_text_from_url(url)


def _analyze_and_set_words(raw_text: str, min_rank: int, max_rank: int) -> bool:
    """Run rank-based analysis and update session state. Returns success."""
    if len(raw_text) <= 2:
        return False
    final_data, raw_count, stats_info = _cached_analyze(raw_text, min_rank, max_rank, False)
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
        result_words = [w.strip() for w in edited_words.split('\n') if w.strip()]
        st.info(f"📝 已编辑：当前共 {len(result_words)} 个单词")
    else:
        result_words = words_only

    # Fix 10: Studied words duplicate detection
    studied = st.session_state.get("studied_words_set", set())
    if studied:
        overlap = [w for w in result_words if w.lower() in studied]
        if overlap:
            preview = ", ".join(overlap[:10]) + ("..." if len(overlap) > 10 else "")
            st.caption(f"📚 {len(overlap)} 个词已在学习记录中：{preview}")
            if st.checkbox("过滤已学词", key="chk_filter_studied"):
                result_words = [w for w in result_words if w.lower() not in studied]

    return result_words


def _render_audio_settings(key_prefix: str) -> tuple[bool, str, bool]:
    """Render audio toggle + voice selector + example audio.
    Returns (enable_audio, voice_code, enable_example_audio)."""
    enable_audio = st.checkbox("启用语音", value=True, key=f"chk_audio_{key_prefix}")
    enable_example_audio = True
    if enable_audio:
        selected_voice_label = st.radio(
            "🎙️ 发音人",
            options=list(constants.VOICE_MAP.keys()),
            index=0,
            horizontal=False,
            key=f"sel_voice_{key_prefix}",
        )
        voice_code = constants.VOICE_MAP[selected_voice_label]
        enable_example_audio = st.checkbox("例句发音", value=True, key=f"chk_example_audio_{key_prefix}", help="为例句生成语音")
    else:
        voice_code = list(constants.VOICE_MAP.values())[0]
    return enable_audio, voice_code, enable_example_audio


def _render_builtin_ai_section(
    words_only: list, enable_audio: bool, voice_code: str, enable_example_audio: bool,
    card_format, use_builtin_ai: str
) -> None:
    """Builtin AI generation flow: button → progress → parse → edit → package → download."""
    ai_model_label = get_config()["openai_model"]

    # Fix 1: Deck name input
    deck_name = st.text_input(
        "🏷️ 牌组名称",
        value=f"Vocab_{get_beijing_time_str()}",
        key="builtin_deck_name",
        help="生成的 .apkg 文件和 Anki 牌组名称",
    )

    # Fix 7: Pagination — slice words into pages of MAX_AUTO_LIMIT
    total_word_count = len(words_only)
    page_size = constants.MAX_AUTO_LIMIT
    page = st.session_state.get("_builtin_gen_page", 0)
    start_idx = page * page_size
    end_idx = (page + 1) * page_size
    words_for_auto_ai = words_only[start_idx:min(end_idx, total_word_count)]
    remaining_words = max(0, total_word_count - end_idx)

    if total_word_count > page_size:
        if page == 0:
            st.caption(
                f"⚠️ 共 {total_word_count} 词，首批处理前 {len(words_for_auto_ai)} 词。"
                " 生成并确认后可继续处理剩余词汇。"
            )
        else:
            st.caption(
                f"⚠️ 第 {page + 1} 批，第 {start_idx + 1}–{min(end_idx, total_word_count)} 词"
                f"（共 {total_word_count} 词）。"
            )

    if use_builtin_ai == "builtin":
        # Handle auto-generate flag set by "继续下一批" button
        auto_generate = st.session_state.pop("_builtin_auto_generate", False)

        col_btn, _ = st.columns([0.35, 1])
        with col_btn:
            clicked = st.button(f"🚀 使用 {ai_model_label} 生成", type="primary", key="btn_builtin_gen")

        if clicked:
            # Fresh generate: reset page and clear previous cards
            st.session_state["_builtin_gen_page"] = 0
            st.session_state.pop("_builtin_parsed_cards", None)
            st.session_state.pop("_builtin_ai_partial_result", None)
            st.session_state.pop("card_editor", None)
            # Re-compute pagination for page 0
            words_for_auto_ai = words_only[:page_size]
            remaining_words = max(0, total_word_count - page_size)

        if clicked or auto_generate:
            batch_allowed, batch_msg = check_batch_limit()
            if not batch_allowed:
                st.warning(batch_msg)
                st.stop()
            record_batch()

            progress_title = st.empty()
            card_text = st.empty()
            card_bar = st.progress(0.0)
            audio_text = st.empty()

            total_words = len(words_for_auto_ai)
            batch_size = constants.AI_BATCH_SIZE
            first_end = min(batch_size, total_words)
            progress_title.markdown("#### ⏳ 内置 AI 制卡进度")
            st.caption("💡 第一组可能稍慢（建立连接），后续组并发进行。")
            card_text.markdown(f"**制卡进度**：第 1 组（1–{first_end}/{total_words}）AI 生成中...")
            audio_text.markdown("**音频进度**：等待制卡完成...")

            def update_ai_progress(current: int, total: int) -> None:
                ratio = current / total if total > 0 else 0.0
                card_bar.progress(min(0.9, ratio * 0.9))
                batch_idx = (current + batch_size - 1) // batch_size
                start = (batch_idx - 1) * batch_size + 1
                end = min(batch_idx * batch_size, total)
                card_text.markdown(f"**制卡进度**：第 {batch_idx} 组（{start}–{end}/{total}）AI 生成中...")

            # Fix 3: Unpack (content, failed_words) tuple
            ai_result, failed_words = process_ai_in_batches(
                words_for_auto_ai,
                progress_callback=update_ai_progress,
                card_format=card_format,
            )

            # Fix 3: Report failed words
            if failed_words:
                st.warning(
                    f"⚠️ {len(failed_words)} 个词生成失败，已跳过："
                    f"{', '.join(failed_words[:20])}"
                    + ("..." if len(failed_words) > 20 else "")
                )

            if ai_result:
                card_text.markdown("**制卡进度**：正在解析 AI 结果...")
                parsed_data = parse_anki_data(ai_result)

                if parsed_data:
                    card_bar.progress(1.0)
                    card_text.markdown(f"**制卡进度**：✅ 完成（共 {len(parsed_data)} 张）")
                    audio_text.markdown("**音频进度**：请在下方确认卡片后开始。")

                    # Fix 4: Checkpoint — save raw AI text in case packaging fails
                    st.session_state["_builtin_ai_partial_result"] = ai_result

                    # Fix 5: Save cards for editing; append if continuing a batch
                    if auto_generate:
                        existing = st.session_state.get("_builtin_parsed_cards") or []
                        st.session_state["_builtin_parsed_cards"] = existing + parsed_data
                        st.session_state.pop("card_editor", None)
                    else:
                        st.session_state["_builtin_parsed_cards"] = parsed_data

                    st.info(
                        f"✅ 解析完成，共 {len(st.session_state['_builtin_parsed_cards'])} 张卡片。"
                        " 请在下方确认并编辑后生成 .apkg。"
                    )
                else:
                    card_text.markdown("**制卡进度**：❌ 解析失败")
                    audio_text.markdown("**音频进度**：未开始")
                    st.error("解析失败，AI 返回内容为空或格式错误。")
            else:
                card_text.markdown("**制卡进度**：❌ AI 生成失败")
                audio_text.markdown("**音频进度**：未开始")
                st.error("AI 生成失败，请检查 API Key 或网络连接。")

        # Fix 5: Card editor — shown whenever cards are in session state
        if st.session_state.get("_builtin_parsed_cards"):
            cards = st.session_state["_builtin_parsed_cards"]

            st.markdown("#### ✏️ 确认并编辑卡片")
            st.caption(f"共 {len(cards)} 张卡片，可直接在下方编辑后点击「确认并生成」。")

            cards_df = pd.DataFrame(cards)
            col_config = {
                "w": st.column_config.TextColumn("单词/短语"),
                "m": st.column_config.TextColumn("释义"),
                "e": st.column_config.TextColumn("例句"),
            }
            if "r" in cards_df.columns:
                col_config["r"] = st.column_config.TextColumn("词源")

            edited_df = st.data_editor(
                cards_df,
                column_config=col_config,
                use_container_width=True,
                num_rows="dynamic",
                key="card_editor",
            )

            # Fix 6: Individual card regeneration
            word_list_for_regen = [c.get("w", "") for c in cards if c.get("w")]
            if word_list_for_regen:
                col_sel, col_regen_btn = st.columns([3, 1])
                with col_sel:
                    regen_word = st.selectbox(
                        "选择要重新生成的词",
                        options=word_list_for_regen,
                        key="regen_word_select",
                    )
                with col_regen_btn:
                    if st.button("🔄 重新生成", key="btn_regen_single"):
                        with st.spinner(f"正在重新生成「{regen_word}」..."):
                            new_result, _ = process_ai_in_batches([regen_word], card_format=card_format)
                            if new_result:
                                new_cards = parse_anki_data(new_result)
                                if new_cards:
                                    current_cards = list(st.session_state["_builtin_parsed_cards"])
                                    idx = next(
                                        (i for i, c in enumerate(current_cards) if c.get("w") == regen_word),
                                        None,
                                    )
                                    if idx is not None:
                                        current_cards[idx] = new_cards[0]
                                        st.session_state["_builtin_parsed_cards"] = current_cards
                                        st.session_state.pop("card_editor", None)
                                        st.toast(f"「{regen_word}」已重新生成", icon="✅")
                                        st.rerun()
                                    else:
                                        st.warning(f"未找到词条「{regen_word}」")
                                else:
                                    st.warning("重新生成失败，AI 返回格式错误。")
                            else:
                                st.warning("重新生成失败，请检查网络连接。")

            # Confirm button + next-batch button
            col_confirm, col_continue = st.columns([2, 1])
            with col_confirm:
                if st.button(
                    "✅ 确认并生成 .apkg", key="btn_confirm_gen",
                    type="primary", use_container_width=True
                ):
                    try:
                        edited_data = edited_df.to_dict("records")
                        st.session_state["_builtin_parsed_cards"] = edited_data
                        st.session_state["anki_cards_cache"] = edited_data  # Fix 10: for mark-as-studied

                        audio_text2 = st.empty()
                        audio_bar2 = st.progress(0.0)
                        audio_text2.markdown("**音频进度**：进行中...")

                        def update_pkg_progress(ratio: float, text: str) -> None:
                            audio_bar2.progress(ratio)
                            audio_text2.markdown(f"**音频进度**：{text}")

                        current_deck_name = st.session_state.get("builtin_deck_name", deck_name)
                        file_path, audio_failed, failed_phrases = generate_anki_package(
                            edited_data,
                            current_deck_name,
                            enable_tts=enable_audio,
                            tts_voice=voice_code,
                            enable_example_tts=enable_example_audio,
                            progress_callback=update_pkg_progress,
                        )
                        set_anki_pkg(file_path, current_deck_name)
                        st.session_state["_builtin_audio_failed_count"] = audio_failed
                        st.session_state["_builtin_audio_failed_phrases"] = failed_phrases or []
                        audio_bar2.progress(1.0)
                        suffix = f"（{audio_failed} 条失败，可点击下方重试）" if audio_failed else ""
                        audio_text2.markdown(f"**音频进度**：✅ 完成{suffix}")
                        st.balloons()
                        run_gc()
                    except Exception as e:
                        ErrorHandler.handle(e, "生成出错")

            # Fix 7: Continue with next batch
            with col_continue:
                if remaining_words > 0:
                    if st.button(
                        f"▶ 下一批（剩余 {remaining_words} 词）",
                        key="btn_gen_next_page",
                        use_container_width=True,
                    ):
                        st.session_state["_builtin_gen_page"] = page + 1
                        st.session_state["_builtin_auto_generate"] = True
                        st.rerun()

    else:
        st.info("请使用右侧「复制 Prompt」到第三方 AI，格式与上方通用设置一致。")

    render_anki_download_button(
        f"📥 下载 {st.session_state.get('anki_pkg_name', 'deck.apkg')}",
        button_type="primary",
        use_container_width=True,
    )

    # Fix 10: Mark as studied after downloading
    if st.session_state.get("anki_pkg_path") and st.session_state.get("anki_cards_cache"):
        if st.button("📚 标记为已学", key="btn_mark_studied_builtin"):
            cards_to_mark = st.session_state.get("anki_cards_cache") or []
            studied = st.session_state.get("studied_words_set", set())
            studied.update(c.get("w", "").lower() for c in cards_to_mark if c.get("w"))
            st.session_state["studied_words_set"] = studied
            st.toast(f"已记录 {len(cards_to_mark)} 个词到学习历史", icon="📚")

    # Audio retry: only re-run TTS for files still missing (cache skips successful ones)
    audio_failed = st.session_state.get("_builtin_audio_failed_count", 0)
    failed_phrases = st.session_state.get("_builtin_audio_failed_phrases", [])
    if audio_failed > 0 and st.session_state.get("_builtin_parsed_cards") and st.session_state.get("anki_pkg_path"):
        st.warning(
            f"⚠️ {audio_failed} 条音频生成失败，涉及 {len(failed_phrases)} 个词。"
            " 点击重试，已成功的音频会直接复用，只补全缺失部分。"
        )
        with st.expander("❓ 音频失败可能原因", expanded=False):
            st.markdown("""
- **网络超时**：edge-tts 需联网，网络不稳或超时会导致失败
- **文本过短**：例句 ≤3 字符可能被 TTS 忽略或生成失败
- **非 ASCII 括号**：例句含中文括号 `（）` 等，TTS 可能收到错误文本
- **速率限制**：大批量时服务端可能限流
- **临时目录**：权限或磁盘空间不足导致写入失败
- **语音不支持**：极少数字符或语言组合不被所选语音支持
""")
        if failed_phrases:
            st.caption("失败词汇：" + "、".join(failed_phrases[:20]) + (" …" if len(failed_phrases) > 20 else ""))
        if st.button("🔄 重试失败音频", key="btn_retry_audio_builtin"):
            cards = st.session_state["_builtin_parsed_cards"]
            with st.spinner(f"⏳ 重试 {audio_failed} 条音频..."):
                try:
                    current_deck_name = st.session_state.get("builtin_deck_name", deck_name)
                    new_file, new_failed, new_phrases = generate_anki_package(
                        cards, current_deck_name,
                        enable_tts=enable_audio, tts_voice=voice_code,
                        enable_example_tts=enable_example_audio,
                    )
                    set_anki_pkg(new_file, current_deck_name)
                    st.session_state["_builtin_audio_failed_count"] = new_failed
                    st.session_state["_builtin_audio_failed_phrases"] = new_phrases or []
                    run_gc()
                    st.rerun()
                except Exception as e:
                    ErrorHandler.handle(e, "重试音频失败")

    st.caption("⚠️ AI 结果请人工复核后再学习。")


def _render_thirdparty_prompt_section(
    words_only: list, examples_colloquial: bool, use_builtin_ai: str,
    enable_audio: bool = True, voice_code: str = "", enable_example_audio: bool = True
) -> None:
    """Render card-format options and a copyable prompt block for third-party AI services."""
    if use_builtin_ai == "thirdparty":
        st.markdown("#### 📌 复制 Prompt（可自定义卡片格式）")
        st.caption("在下方选择卡片格式后复制 Prompt 到第三方 AI，生成结果粘贴到「Anki 制卡」页解析。")
    else:
        st.markdown("#### 第三方 AI Prompt")
        st.caption("需要大批量或自定义卡片格式时，可切换为「第三方 AI」在下方自定义格式并复制 Prompt。")

    with st.expander("📌 复制 Prompt（第三方 AI）", expanded=(use_builtin_ai == "thirdparty")):
        st.markdown("##### ⚙️ 自定义卡片格式")
        st.caption("仅影响下方生成的 Prompt")
        st.markdown("**卡片内容**")
        r1a, r1b = st.columns(2)
        with r1a:
            tp_front = st.selectbox(
                "正面",
                options=["word", "phrase"],
                format_func=lambda x: "单词" if x == "word" else "短语/词组",
                index=1,
                key="tp_prompt_front",
            )
        with r1b:
            tp_def = st.selectbox(
                "释义",
                options=["cn", "en", "en_native", "both"],
                format_func=lambda x: {
                    "cn": "中文",
                    "en": "英文（学习型词典）",
                    "en_native": "英文（母语者词典）",
                    "both": "中英双语",
                }[x],
                index=2,
                key="tp_prompt_def",
            )
        r2a, r2b = st.columns(2)
        with r2a:
            tp_ex = st.selectbox(
                "例句数量",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} 条",
                index=2,
                key="tp_prompt_ex",
            )
        with r2b:
            tp_ety = st.checkbox("词根词源词缀", value=True, key="tp_prompt_ety")
        st.markdown("**例句样式**")
        r3a, r3b = st.columns(2)
        with r3a:
            tp_ex_cn = st.checkbox("例句带中文翻译", value=False, key="tp_prompt_ex_cn")
        with r3b:
            tp_colloquial = st.checkbox("例句用口语", value=examples_colloquial, key="tp_prompt_colloquial", help="例句使用日常口语化表达")

        st.markdown("---")
        st.markdown("**分组与 Prompt**")
        third_party_card_format: CardFormat = {
            "front": tp_front,
            "definition": tp_def,
            "examples": tp_ex,
            "examples_with_cn": tp_ex_cn,
            "etymology": tp_ety,
            "examples_colloquial": tp_colloquial,
        }

        st.caption("选择分组并复制下方 Prompt 到第三方 AI")
        batch_size_prompt = int(
            st.number_input(f"🔢 分组大小 (最大 {constants.MAX_AUTO_LIMIT})", min_value=1, max_value=constants.MAX_AUTO_LIMIT, value=50, step=10, key="batch_size_prompt")
        )
        current_batch_words = []

        if words_only:
            total_w = len(words_only)
            if total_w <= constants.MAX_AUTO_LIMIT:
                st.caption(f"💡 当前共 {total_w} 个单词（≤{constants.MAX_AUTO_LIMIT}），已全部放入一个 Prompt。")
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
        render_prompt_copy_button(strict_prompt_template, key="copy_tp_prompt")
        st.code(strict_prompt_template, language="text")

    st.caption("💡 复制 Prompt 到第三方 AI 后，请到「anki制卡」页粘贴结果并生成 .apkg。")

    render_anki_download_button(
        f"📥 下载 {st.session_state.get('anki_pkg_name', 'deck.apkg')}",
        button_type="primary",
        use_container_width=True,
    )
    st.caption("⚠️ AI 结果请人工复核后再学习。")


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
    enable_audio, voice_code, enable_example_audio = _render_audio_settings("auto")
    examples_colloquial = st.checkbox(
        "例句用口语",
        value=False,
        key="chk_examples_colloquial",
        help="例句使用日常口语化表达，而非书面语",
    )

    # Fix 2: Configurable card format for built-in AI
    with st.expander("⚙️ 卡片格式（内置 AI）", expanded=False):
        st.caption("仅影响「内置 AI 一键生成」；第三方 AI Prompt 在下方单独设置。")
        _cf_col_a, _cf_col_b = st.columns(2)
        with _cf_col_a:
            builtin_front = st.selectbox(
                "正面",
                options=["word", "phrase"],
                format_func=lambda x: "单词" if x == "word" else "短语/词组",
                index=0,
                key="builtin_front",
            )
            builtin_def = st.selectbox(
                "释义",
                options=["cn", "en", "en_native", "both"],
                format_func=lambda x: {
                    "cn": "中文",
                    "en": "英文（学习型词典）",
                    "en_native": "英文（母语者词典）",
                    "both": "中英双语",
                }[x],
                index=3,
                key="builtin_def",
            )
        with _cf_col_b:
            builtin_ex = st.selectbox(
                "例句数量",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} 条",
                index=2,
                key="builtin_ex",
            )
            builtin_ety = st.checkbox("词根词源", value=True, key="builtin_ety")
        builtin_ex_cn = st.checkbox("例句带中文翻译", value=True, key="builtin_ex_cn")

    shared_card_format: CardFormat = {
        "front": builtin_front,
        "definition": builtin_def,
        "examples": builtin_ex,
        "examples_with_cn": builtin_ex_cn,
        "etymology": builtin_ety,
        "examples_colloquial": examples_colloquial,
    }

    st.markdown("---")
    st.markdown("#### ② 生成方式")
    st.markdown('<div id="ai-gen-mode-radio"></div>', unsafe_allow_html=True)
    use_builtin_ai = st.radio(
        "选择",
        options=["builtin", "thirdparty"],
        format_func=lambda x: "内置 AI 一键生成" if x == "builtin" else "第三方 AI（复制 Prompt）",
        index=0,
        horizontal=True,
        key="ai_gen_mode",
    )
    st.markdown("")  # 间距

    if use_builtin_ai == "thirdparty":
        # 第三方 AI 选中时：全宽展示，充分利用空间
        _render_thirdparty_prompt_section(words_only, examples_colloquial, use_builtin_ai, enable_audio, voice_code, enable_example_audio)
    else:
        # 内置 AI 时：不显示折叠栏，仅展示一键生成区
        _render_builtin_ai_section(words_only, enable_audio, voice_code, enable_example_audio, shared_card_format, use_builtin_ai)


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

    _btn_label = "查询中..." if st.session_state["quick_lookup_is_loading"] else f"🔍 {get_config()['openai_model']}"
    _has_content = bool(st.session_state.get("quick_lookup_word") or st.session_state.get("quick_lookup_last_result"))

    with st.form("quick_lookup_form", clear_on_submit=False, border=False):
        # Always render 3 columns so the form structure stays constant across
        # reruns.  A changing column count (2 vs 3) shifts the submit button's
        # element ID, causing Streamlit to lose the "clicked" signal and making
        # the lookup button silently do nothing on the first submission.
        col_word, col_btn, col_clear = st.columns([4, 2, 1.2])
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
        with col_clear:
            clear_submit = st.form_submit_button("清空", use_container_width=True, disabled=not _has_content)
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

# Fix 9: Session save/restore
with st.expander("💾 存档 / 📂 读取进度", expanded=False):
    col_save, col_load = st.columns(2)
    with col_save:
        st.caption("导出当前单词列表、卡片缓存和学习记录。")
        json_blob = export_session_json()
        st.download_button(
            "💾 导出当前进度",
            data=json_blob,
            file_name=f"vocabflow_{get_beijing_time_str()}.json",
            mime="application/json",
            key="btn_export_session",
        )
    with col_load:
        st.caption("导入之前导出的 .json 存档文件。")
        uploaded_session = st.file_uploader(
            "📂 导入存档文件", type=["json"], key="session_import"
        )
        if uploaded_session is not None:
            try:
                import_session_json(uploaded_session.read().decode())
                st.success("✅ 已恢复进度")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 存档文件格式错误：{e}")

tab_lookup, tab_extract, tab_anki = st.tabs([
    "AI查词",
    "筛选单词",
    "anki制卡",
])

with tab_lookup:
    render_quick_lookup()

# ==========================================
# Tab 1: Word Extraction（筛选单词）
# ==========================================
def _render_shared_rank_selection() -> tuple[int, int]:
    """渲染通用词汇量 rank 选择（文本/链接/文件/词库共用，词表不适用）。返回 (min_rank, max_rank)。"""
    st.markdown("#### 词汇量 rank 选择（文本 / 链接 / 文件 / 词库 通用）")
    preset_choices = [f"{label} ({min_r}–{max_r})" for label, min_r, max_r in constants.RANK_PRESETS] + ["自定义"]
    selected = st.radio(
        "词汇量区间",
        preset_choices,
        key="extract_rank_preset",
        horizontal=True,
        format_func=lambda x: x,
    )
    if selected != "自定义":
        for _label, min_r, max_r in constants.RANK_PRESETS:
            if selected == f"{_label} ({min_r}–{max_r})":
                st.session_state["extract_min_rank"] = min_r
                st.session_state["extract_max_rank"] = max_r
                break
        min_rank = st.session_state["extract_min_rank"]
        max_rank = st.session_state["extract_max_rank"]
    else:
        col1, col2 = st.columns(2)
        min_rank = col1.number_input(
            "忽略前 N 高频词 (Min Rank)", 1, 20000, st.session_state["extract_min_rank"], step=100,
            key="extract_min_rank"
        )
        max_rank = col2.number_input(
            "忽略后 N 低频词 (Max Rank)", 2000, 50000, st.session_state["extract_max_rank"], step=500,
            key="extract_max_rank"
        )
        if max_rank < min_rank:
            st.warning("⚠️ Max Rank 小于 Min Rank，已自动交换两者。")
            min_rank, max_rank = max_rank, min_rank
            st.session_state["extract_min_rank"] = min_rank
            st.session_state["extract_max_rank"] = max_rank
    return min_rank, max_rank


with tab_extract:
    # 通用 rank 区间（文本/链接/文件/词库共用；词表不筛 rank）
    shared_min_rank, shared_max_rank = _render_shared_rank_selection()
    st.markdown("---")

    mode_paste, mode_url, mode_upload, mode_rank, mode_manual = st.tabs([
        "文本",
        "链接",
        "文件",
        "词库",
        "词表",
    ])

    with mode_paste:
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
            if shared_max_rank < shared_min_rank:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请在上方修正后重试。")
            elif len(pasted_text) > constants.MAX_PASTE_TEXT_LENGTH:
                st.error(f"❌ 文本过长（最大约 {constants.MAX_PASTE_TEXT_LENGTH // 1000}K 字符），请缩短后重试。")
            else:
                with st.status("🔍 正在加载资源并分析文本...", expanded=True) as status:
                    start_time = time.time()
                    raw_text = pasted_text

                    status.write("🧠 正在进行 NLP 词形还原与分级...")
                    if _analyze_and_set_words(raw_text, shared_min_rank, shared_max_rank):
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        status.update(label="✅ 分析完成", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 内容为空或太短", state="error")

    with mode_url:
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
            if shared_max_rank < shared_min_rank:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请在上方修正后重试。")
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
                        raw_text = _cached_extract_url(input_url)
                        if raw_text.startswith("Error:"):
                            st.error(f"❌ {raw_text}")
                            status.update(label="❌ 抓取失败", state="error", expanded=False)
                        elif _analyze_and_set_words(raw_text, shared_min_rank, shared_max_rank):
                            st.session_state['process_time'] = time.time() - start_time
                            run_gc()
                            status.update(label="✅ 生成完成", state="complete", expanded=False)
                        else:
                            status.update(label="⚠️ 抓取内容为空或过短", state="error")

    with mode_upload:
        uploaded_file = st.file_uploader(
            "上传文件（TXT/PDF/DOCX/EPUB/CSV/Excel/DB）",
            type=['txt', 'pdf', 'docx', 'epub', 'csv', 'xlsx', 'xls', 'db', 'sqlite'],
            key="upload_2_3",
        )
        if uploaded_file and is_upload_too_large(uploaded_file):
            st.error(f"❌ 文件过大，已限制为 {constants.MAX_UPLOAD_MB}MB。请缩小文件后重试。")
            uploaded_file = None

        if st.button("📁 从文件生成重点词", type="primary", key="btn_mode_2_3"):
            if shared_max_rank < shared_min_rank:
                st.error("❌ Max Rank 必须大于等于 Min Rank，请在上方修正后重试。")
            elif uploaded_file is None:
                st.warning("⚠️ 请先上传文件。")
            else:
                with st.status("📄 正在解析文件并提取重点词...", expanded=True) as status:
                    start_time = time.time()
                    raw_text = extract_text_from_file(uploaded_file)
                    if _analyze_and_set_words(raw_text, shared_min_rank, shared_max_rank):
                        st.session_state['process_time'] = time.time() - start_time
                        run_gc()
                        status.update(label="✅ 生成完成", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ 文件内容为空或过短", state="error")

    with mode_manual:
        st.markdown("#### 直接粘贴词表（不做 rank 筛选）")
        st.caption("支持任意格式：可直接粘贴文章、列表、带序号或符号的文本，将自动提取其中所有英文单词。")
        manual_words_text = st.text_area(
            "✍️ 粘贴任意包含英文单词的文本",
            height=220,
            key="manual_words_2_5",
            placeholder="如：1. altruism 2. hectic  或 直接粘贴整段英文…",
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
                    # 弱化格式要求：从整段文本中提取所有英文单词，自动去除符号、空格等
                    valid_words = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", manual_words_text)
                    valid_words = [
                        w for w in valid_words
                        if constants.MIN_WORD_LENGTH <= len(w) <= constants.MAX_WORD_LENGTH
                    ]

                    if not valid_words:
                        st.error("❌ 没有识别到有效的英文单词（2–25 字母），请检查输入内容。")
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
        st.caption("使用上方「词汇量 rank 选择」的区间；顺序生成从区间起点起取，随机抽取在区间内随机取。")
        gen_type = st.radio("生成模式", ["🔢 顺序生成", "🔀 随机抽取"], horizontal=True)

        if "顺序生成" in gen_type:
            # 当 rank 选择变化时，起始排名跟随 shared_min_rank 更新
            if st.session_state.get("rank_start_sync_min") != shared_min_rank:
                st.session_state["rank_start_2_4"] = shared_min_rank
                st.session_state["rank_start_sync_min"] = shared_min_rank
            col_a, col_b = st.columns(2)
            start_rank = col_a.number_input("起始排名", 1, 20000, shared_min_rank, step=100, key="rank_start_2_4")
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
            random_count = st.number_input("抽取数量", 10, 5000, 10, step=10, key="rank_random_count_2_4")
            st.caption(f"当前区间：{shared_min_rank} – {shared_max_rank}（在上方修改）")

            if st.button("🎲 随机抽取"):
                if shared_max_rank < shared_min_rank:
                    st.error("❌ 最大排名必须大于等于最小排名，请在上方修正后重试。")
                else:
                    with st.spinner("正在抽取..."):
                        if FULL_DF is not None:
                            rank_col = next((c for c in FULL_DF.columns if 'rank' in c), None)
                            word_col = next((c for c in FULL_DF.columns if 'word' in c), None)
                            if rank_col is None or word_col is None:
                                st.error("❌ 词库CSV格式异常：缺少 rank 或 word 列")
                            else:
                                pool = FULL_DF[(FULL_DF[rank_col] >= shared_min_rank) & (FULL_DF[rank_col] <= shared_max_rank)]
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
        horizontal=False,
        key="sel_voice_manual",
    )
    manual_voice_code = constants.VOICE_MAP[manual_voice_label]

    enable_audio = st.checkbox("启用语音", value=True, key="chk_audio_manual")
    manual_enable_example_audio = st.checkbox("例句发音", value=True, key="chk_example_audio_manual", help="为例句生成语音")

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
                        file_path, manual_audio_failed, manual_failed_phrases = generate_anki_package(
                            parsed_data,
                            deck_name,
                            enable_tts=enable_audio,
                            tts_voice=manual_voice_code,
                            enable_example_tts=manual_enable_example_audio,
                            progress_callback=update_progress_manual
                        )

                        set_anki_pkg(file_path, deck_name)
                        st.session_state["_manual_audio_failed_count"] = manual_audio_failed
                        st.session_state["_manual_audio_failed_phrases"] = manual_failed_phrases or []

                        done_msg = f"✅ **生成完毕！共制作 {len(parsed_data)} 张卡片**"
                        if manual_audio_failed:
                            done_msg += f"（{manual_audio_failed} 条音频失败，可点「重试失败音频」）"
                        status_manual.markdown(done_msg)
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

        # Fix 10: Mark as studied after downloading
        if st.session_state.get("anki_pkg_path"):
            if st.button("📚 标记为已学", key="btn_mark_studied_anki"):
                cards_to_mark = st.session_state.get("anki_cards_cache") or []
                studied = st.session_state.get("studied_words_set", set())
                studied.update(c.get("w", "").lower() for c in cards_to_mark if c.get("w"))
                st.session_state["studied_words_set"] = studied
                st.toast(f"已记录 {len(cards_to_mark)} 个词到学习历史", icon="📚")

        # Audio retry for manual tab
        manual_audio_failed = st.session_state.get("_manual_audio_failed_count", 0)
        manual_failed_phrases = st.session_state.get("_manual_audio_failed_phrases", [])
        if manual_audio_failed > 0 and st.session_state.get("anki_cards_cache") and st.session_state.get("anki_pkg_path"):
            st.warning(f"⚠️ {manual_audio_failed} 条音频生成失败，涉及 {len(manual_failed_phrases)} 个词。点击重试（已成功的音频将直接复用）。")
            with st.expander("❓ 音频失败可能原因", expanded=False):
                st.markdown("""
- **网络超时**：edge-tts 需联网，网络不稳或超时会导致失败
- **文本过短**：例句 ≤3 字符可能被 TTS 忽略或生成失败
- **非 ASCII 括号**：例句含中文括号 `（）` 等，TTS 可能收到错误文本
- **速率限制**：大批量时服务端可能限流
- **临时目录**：权限或磁盘空间不足导致写入失败
- **语音不支持**：极少数字符或语言组合不被所选语音支持
""")
            if manual_failed_phrases:
                st.caption("失败词汇：" + "、".join(manual_failed_phrases[:20]) + (" …" if len(manual_failed_phrases) > 20 else ""))
            if st.button("🔄 重试失败音频", key="btn_retry_audio_manual"):
                _retry_cards = st.session_state["anki_cards_cache"]
                _retry_deck = st.session_state.get("anki_pkg_name", "deck.apkg").removesuffix(".apkg")
                with st.spinner(f"⏳ 重试 {manual_audio_failed} 条音频..."):
                    try:
                        _retry_voice = constants.VOICE_MAP.get(
                            st.session_state.get("sel_voice_manual", ""),
                            list(constants.VOICE_MAP.values())[0]
                        )
                        _retry_file, _retry_failed, _retry_phrases = generate_anki_package(
                            _retry_cards, _retry_deck,
                            enable_tts=st.session_state.get("chk_audio_manual", True),
                            tts_voice=_retry_voice,
                            enable_example_tts=st.session_state.get("chk_example_audio_manual", True),
                        )
                        set_anki_pkg(_retry_file, _retry_deck)
                        st.session_state["_manual_audio_failed_count"] = _retry_failed
                        st.session_state["_manual_audio_failed_phrases"] = _retry_phrases or []
                        run_gc()
                        st.rerun()
                    except Exception as e:
                        ErrorHandler.handle(e, "重试音频失败")

st.markdown(
    '<div class="app-footer">Vocab Flow Ultra &nbsp;·&nbsp; Built for learners</div>',
    unsafe_allow_html=True
)
