# app.py — 完整程序
# 依赖: streamlit, genanki
# pip install streamlit genanki

import streamlit as st
import json
import re
import random
import hashlib
import io
import os
from typing import List, Dict, Any

try:
    import genanki
except Exception as e:
    st = None  # avoid reference error when viewing file outside Streamlit
    raise

# --------------------------
# Utilities: JSON robust parser
# --------------------------

def normalize_json_like_string(s: str) -> str:
    """
    轻度规范化非标准 JSON（智能引号、单引号）作为 fallback。
    尽量保守，不要过度修改以免破坏合法内容。
    """
    if not isinstance(s, str):
        return s
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    # 如果文本中没有双引号，但有单引号，尝试替换为双引号（fallback）
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    return s

def safe_json_loads(s: str):
    """
    先严格尝试 json.loads，失败后尝试轻度规范化再解析。
    抛异常交由调用方处理。
    """
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(normalize_json_like_string(s))
        except Exception:
            raise

def extract_json_objects_from_text(text: str) -> List[str]:
    """
    基于花括号深度和字符串转义状态提取出文本中的所有完整 JSON 对象或嵌套对象。
    更稳健地处理跨行、字符串内部的花括号等情况。
    返回 JSON 字符串列表（每项为"{...}"）。
    """
    results = []
    in_string = False
    escape = False
    depth = 0
    start_idx = None

    for i, ch in enumerate(text):
        # 处理转义
        if ch == "\\" and in_string:
            escape = not escape
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        else:
            escape = False

        if not in_string:
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        results.append(text[start_idx:i+1])
                        start_idx = None
    return results

def map_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    将各种常见字段名映射到标准键：
      front, meaning, examples, etymology
    支持常见别名（w, word, front, phrase / m, meaning, def / e, examples ...）
    """
    if not isinstance(d, dict):
        return {}

    def pick(keys):
        for k in keys:
            if k in d:
                return d[k]
        return ""

    mapped = {
        "front": pick(["w", "word", "front", "phrase", "front_phrase", "text"]),
        "meaning": pick(["m", "meaning", "definition", "def"]),
        "examples": pick(["e", "examples", "example", "ex", "sentences"]),
        "etymology": pick(["r", "etymology", "root", "词源"]),
    }
    # 保证所有键存在
    for k in ["front", "meaning", "examples", "etymology"]:
        mapped.setdefault(k, "")
    return mapped

def normalize_values(card: Dict[str, Any]) -> Dict[str, str]:
    """
    统一字段值类型：
    - front/meaning/etymology -> str (strip)
    - examples -> 如果是 list, join 为 <br>；如果是 str，替换换行符为 <br>
    """
    out = {}
    for k in ["front", "meaning", "etymology"]:
        v = card.get(k, "")
        if v is None:
            out[k] = ""
        else:
            out[k] = str(v).strip()

    ex = card.get("examples", "")
    if ex is None:
        out["examples"] = ""
    elif isinstance(ex, (list, tuple)):
        out["examples"] = "<br>".join([str(x).strip() for x in ex if x is not None and str(x).strip()])
    else:
        out["examples"] = str(ex).strip().replace("\r\n", "\n").replace("\n", "<br>")

    return out

def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """
    从任意 AI 输出文本中解析出卡片列表（支持：JSON array, NDJSON, 单个 JSON, 混杂文本）
    返回: list of dict with keys: front, meaning, examples, etymology
    """
    if not raw_text or not raw_text.strip():
        return []

    # 移除 markdown code fence 标记（例如 ```json ... ```）
    text = raw_text.replace("```json", "").replace("```", "").strip()

    candidate_objects: List[Any] = []

    # 1) 尝试整体 JSON 解析（array 或 object）
    try:
        loaded = safe_json_loads(text)
        if isinstance(loaded, list):
            candidate_objects.extend(loaded)
        elif isinstance(loaded, dict):
            candidate_objects.append(loaded)
    except Exception:
        # 忽略，进入下一种解析策略
        pass

    # 2) 如果为空，尝试按行解析（NDJSON）
    if not candidate_objects:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 处理 "1. {" 或 "1) {" 前缀
            if re.match(r'^\d+[\.\)]\s*[{]', line):
                line = re.sub(r'^\d+[\.\)]\s*', '', line, count=1)
            if line.startswith("{") and line.endswith("}"):
                try:
                    candidate_objects.append(safe_json_loads(line))
                    continue
                except Exception:
                    # 行可能被截断或包含非标准 json，跳过
                    continue

    # 3) 如果仍然没有，使用花括号平衡法提取所有 JSON 块
    if not candidate_objects:
        chunks = extract_json_objects_from_text(text)
        for chunk in chunks:
            try:
                candidate_objects.append(safe_json_loads(chunk))
            except Exception:
                # 尝试轻度替换后解析
                try:
                    candidate_objects.append(json.loads(normalize_json_like_string(chunk)))
                except Exception:
                    # 最终忽略不可解析的 chunk
                    continue

    # 处理 candidate_objects，映射字段并去重
    cards: List[Dict[str, str]] = []
    seen = set()

    for obj in candidate_objects:
        if not isinstance(obj, dict):
            continue
        mapped = map_fields(obj)
        norm = normalize_values(mapped)

        front = (norm.get("front") or "").strip()
        meaning = (norm.get("meaning") or "").strip()

        # 最低要求：front 和 meaning 必须存在
        if not front or not meaning:
            continue

        key = front.lower()
        if key in seen:
            continue
        seen.add(key)

        # 清理 markdown 强调等常见噪声
        front = re.sub(r'\*\*|__|\*|`', '', front)

        cards.append({
            "front": front,
            "meaning": meaning,
            "examples": norm.get("examples", ""),
            "etymology": norm.get("etymology", "")
        })

    return cards

# --------------------------
# Anki deck & model helpers
# --------------------------

def stable_id_from_string(s: str, mask_bits: int = 31) -> int:
    """生成稳定的正整数 id（用于 model/deck id），基于 SHA1 的前若干位"""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    # 取前 8 字节 -> int
    val = int(h[:16], 16)
    mask = (1 << mask_bits) - 1
    return val & mask

def create_deck_and_package(cards: List[Dict[str, str]], deck_name: str = "AI Deck") -> bytes:
    """
    使用 genanki 创建 deck，并返回写好的 .apkg 二进制内容（bytes）
    """
    if not cards:
        raise ValueError("No cards to create deck from.")

    # 生成稳定 model id & deck id
    model_id = stable_id_from_string("model:" + deck_name)
    deck_id = stable_id_from_string("deck:" + deck_name)

    model = genanki.Model(
        model_id,
        "AI Model",
        fields=[
            {"name": "Front"},
            {"name": "Meaning"},
            {"name": "Examples"},
            {"name": "Etymology"},
        ],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": """{{FrontSide}}
<hr>
<b>Meaning:</b><br>{{Meaning}}<br><br>
<b>Examples:</b><br>{{Examples}}<br><br>
<b>Etymology:</b><br>{{Etymology}}"""
        }],
        css="""
.card {
  font-family: arial;
  font-size: 20px;
  text-align: left;
  color: #333;
  background-color: white;
}
"""
    )

    deck = genanki.Deck(deck_id, deck_name)

    for c in cards:
        note = genanki.Note(
            model=model,
            fields=[c["front"], c["meaning"], c["examples"], c["etymology"]],
            guid=None
        )
        deck.add_note(note)

    pkg = genanki.Package(deck)
    # 写入内存 bytes
    buf = io.BytesIO()
    pkg.write_to_file("temp_output.apkg")  # genanki 仅支持写文件路径 -> 写临时文件再读回来
    with open("temp_output.apkg", "rb") as f:
        data = f.read()
    try:
        os.remove("temp_output.apkg")
    except Exception:
        pass
    return data

# --------------------------
# Streamlit App UI
# --------------------------

def main():
    st.set_page_config(page_title="AI → Anki (.apkg) 帮手", layout="wide")
    st.title("AI 输出 → Anki `.apkg` 生成器（完整程序）")
    st.markdown(
        "粘贴 AI 返回的 JSON / NDJSON / 混合文本，或上传包含 JSON 的文件，程序会尽力解析并生成 `.apkg`。"
    )

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("输入：AI 输出")
        raw = st.text_area("将 AI 的输出粘到这里（支持多种格式）", height=350, key="raw_input")

        uploaded_file = st.file_uploader("或上传包含 AI 输出的文件（.json/.txt/.ndjson）", type=["json","txt","ndjson"])
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()
                try:
                    raw_text_from_file = raw_bytes.decode("utf-8")
                except Exception:
                    raw_text_from_file = raw_bytes.decode("latin-1")
                # 如果用户同时粘贴和上传，以上传为准
                raw = raw_text_from_file
                st.success("已加载上传文件内容。")
            except Exception as e:
                st.error(f"读取上传文件失败: {e}")

        st.markdown("**提示**：支持以下格式示例：NDJSON（每行一个 JSON）、JSON 数组、单个对象、代码块包裹（```json ... ```）、单引号等常见不规范输出。")

    with col2:
        st.subheader("生成设置")
        deck_name = st.text_input("Deck 名称", value="AI Cards")
        dedupe = st.checkbox("去重（按 front 字段，不区分大小写）", value=True)
        require_meaning = st.checkbox("必须包含 meaning 字段（否则跳过）", value=True)
        sample_btn = st.button("插入示例 JSON")

        if sample_btn:
            sample = """[
  {"w":"apple","m":"苹果","e":["I ate an apple.","The apple is red."]},
  {"w":"run","m":"跑步","e":"I run every day."}
]
"""
            st.session_state["raw_input"] = sample
            st.experimental_rerun()

    # 解析（无论来自粘贴或上传）
    cards = parse_anki_data(raw or "")

    # 额外过滤：按用户设置
    if require_meaning:
        cards = [c for c in cards if c.get("meaning")]

    if dedupe:
        seen = set()
        deduped = []
        for c in cards:
            key = c["front"].strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(c)
        cards = deduped

    st.markdown("---")
    st.subheader("解析结果预览")
    if not cards:
        st.info("尚未解析到有效卡片。提示：可以粘贴示例或上传文件测试。")
    else:
        st.write(f"已解析出 **{len(cards)}** 张卡片（展示前 200 字）")
        # 显示为表格（只显示部分字段）
        from html import escape
        def make_row_html(idx, card):
            return f"""
            <tr>
              <td style="vertical-align:top">{idx+1}</td>
              <td style="vertical-align:top"><b>{escape(card['front'])}</b></td>
              <td style="vertical-align:top">{escape(card['meaning'])}</td>
              <td style="vertical-align:top">{escape(card['examples'][:200])}</td>
            </tr>
            """
        rows = "".join(make_row_html(i, c) for i, c in enumerate(cards))
        table_html = f"""
        <table style="width:100%; border-collapse:collapse">
          <thead><tr><th>#</th><th>Front</th><th>Meaning</th><th>Examples (truncated)</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("生成 .apkg 并下载"):
        if not cards:
            st.error("无法生成：当前没有有效卡片。")
        else:
            try:
                apkg_bytes = create_deck_and_package(cards, deck_name=deck_name)
                st.success(f"已生成 {len(cards)} 张卡片的 .apkg")
                st.download_button("点击下载 .apkg", data=apkg_bytes, file_name=f"{deck_name}.apkg", mime="application/vnd.android.package-archive")
            except Exception as e:
                st.error(f"生成 .apkg 出错: {e}")

    st.markdown("---")
    st.subheader("常见问题 & 调试")
    st.markdown("""
    - 如果 AI 输出非常不规范（大量文本混杂），建议先在 Prompt 中要求：
      1. 使用标准 JSON（双引号）
      2. 或输出 NDJSON（每行一个 JSON）
      3. 或输出一个 JSON array `[...]`
    - 本程序会尽力解析单引号与智能引号，但不保证 100% 正确。遇到解析失败的特殊文本，请把原始输出粘过来我可以帮你 debug。
    - 如果需要：我可以把解析失败的 chunks 写入日志文件以便人工分析（当前版本在内存中处理）。
    """)

if __name__ == "__main__":
    main()
