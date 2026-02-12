import streamlit as st
import json
import re
import random
import genanki

# =========================
# AI JSON 解析工具
# =========================

def normalize_json_like_string(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    return s


def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except:
        try:
            return json.loads(normalize_json_like_string(s))
        except:
            raise


def extract_json_objects_from_text(text: str):
    results = []
    in_string = False
    escape = False
    depth = 0
    start = None

    for i, ch in enumerate(text):
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
                    start = i
                depth += 1

            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    results.append(text[start:i+1])
                    start = None

    return results


def map_fields(d: dict):
    def pick(keys):
        for k in keys:
            if k in d:
                return d[k]
        return ""

    return {
        "front": pick(["w", "word", "front", "phrase", "text"]),
        "meaning": pick(["m", "meaning", "definition"]),
        "examples": pick(["e", "examples", "example"]),
        "etymology": pick(["r", "etymology", "root"])
    }


def normalize_values(card: dict):
    for k in ["front", "meaning", "etymology"]:
        v = card[k]
        card[k] = "" if v is None else str(v).strip()

    ex = card["examples"]

    if isinstance(ex, list):
        card["examples"] = "<br>".join(str(x) for x in ex if x)
    else:
        card["examples"] = "" if ex is None else str(ex).replace("\n", "<br>").strip()

    return card


def parse_anki_data(raw_text):
    if not raw_text.strip():
        return []

    text = raw_text.replace("```json", "").replace("```", "").strip()

    objects = []

    # 尝试整体 JSON
    try:
        data = safe_json_loads(text)

        if isinstance(data, list):
            objects.extend(data)
        elif isinstance(data, dict):
            objects.append(data)

    except:
        pass

    # 按行 NDJSON
    if not objects:
        for line in text.splitlines():
            line = line.strip()

            if re.match(r'^\d+[\.\)]', line):
                line = re.sub(r'^\d+[\.\)]\s*', '', line)

            if line.startswith("{") and line.endswith("}"):
                try:
                    objects.append(safe_json_loads(line))
                except:
                    continue

    # 花括号提取
    if not objects:
        chunks = extract_json_objects_from_text(text)

        for c in chunks:
            try:
                objects.append(safe_json_loads(c))
            except:
                continue

    cards = []
    seen = set()

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        card = normalize_values(map_fields(obj))

        if not card["front"] or not card["meaning"]:
            continue

        key = card["front"].lower()

        if key in seen:
            continue

        seen.add(key)
        cards.append(card)

    return cards


# =========================
# Anki Deck 生成
# =========================

def create_deck(cards, deck_name="AI Cards"):
    model = genanki.Model(
        random.randrange(1 << 30),
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
            "afmt": """
            {{FrontSide}}
            <hr>
            <b>Meaning:</b><br>{{Meaning}}<br><br>
            <b>Examples:</b><br>{{Examples}}<br><br>
            <b>Etymology:</b><br>{{Etymology}}
            """,
        }]
    )

    deck = genanki.Deck(random.randrange(1 << 30), deck_name)

    for c in cards:
        note = genanki.Note(
            model=model,
            fields=[
                c["front"],
                c["meaning"],
                c["examples"],
                c["etymology"],
            ]
        )
        deck.add_note(note)

    pkg = genanki.Package(deck)
    file_path = "output.apkg"
    pkg.write_to_file(file_path)

    return file_path


# =========================
# Streamlit UI
# =========================

st.title("AI JSON → Anki Deck")

st.write("粘贴 AI 输出的 JSON / NDJSON / 混合文本")

raw = st.text_area("AI 输出", height=300)

if st.button("生成 Anki Deck"):
    cards = parse_anki_data(raw)

    if not cards:
        st.error("❌ 没有解析到有效卡片")
    else:
        st.success(f"✅ 解析成功：{len(cards)} 张卡片")

        file_path = create_deck(cards)

        with open(file_path, "rb") as f:
            st.download_button(
                "下载 .apkg",
                f,
                file_name="anki_cards.apkg"
            )
