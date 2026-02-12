import streamlit as st
import spacy
import pandas as pd
import json
import re
import random
import tempfile
import os
import genanki

# 文件处理库
import pypdf
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# 引入样式和字典
from utils import VOCAB_DICT
import styles

# ==========================================
# 1. NLP 模型加载
# ==========================================
@st.cache_resource
def load_nlp():
    """加载 Spacy 模型"""
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])

nlp = load_nlp()

# ==========================================
# 2. 文件内容提取
# ==========================================
def extract_text_from_file(uploaded_file):
    if not uploaded_file:
        return ""
        
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if file_type == 'pdf':
            reader = pypdf.PdfReader(tmp_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        
        elif file_type in ['docx', 'doc']:
            doc = docx.Document(tmp_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        elif file_type == 'epub':
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
        
        elif file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    except Exception as e:
        st.error(f"解析文件出错: {e}")
        return ""
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return text

# ==========================================
# 3. 核心分析逻辑
# ==========================================
def analyze_logic(raw_text, min_rank, max_rank, include_unknown):
    doc = nlp(raw_text)
    
    candidates = [
        token.lemma_.lower() for token in doc 
        if token.is_alpha and not token.is_stop and len(token.text) > 1
    ]
    
    raw_count = len(doc)
    unique_candidates = list(set(candidates))
    filtered_list = []
    
    for word in unique_candidates:
        rank = VOCAB_DICT.get(word, None)
        if rank is not None:
            if rank >= min_rank and rank <= max_rank:
                filtered_list.append((word, rank))
        else:
            if include_unknown:
                filtered_list.append((word, 999999))
    
    filtered_list.sort(key=lambda x: x[1])
    return filtered_list, raw_count

# ==========================================
# 4. 生成 AI Prompt
# ==========================================
def get_ai_prompt(word_list, front_mode, def_mode, ex_count, need_ety):
    words_str = ", ".join(word_list)
    
    phrase_instr = "Find a common short phrase/collocation using this word." if "Phrase" in front_mode else "Use the word itself."
    lang_instr = "English (Simple definition)"
    if def_mode == "中文": lang_instr = "Chinese only"
    elif def_mode == "中英双语": lang_instr = "Bilingual (Chinese & English)"
    
    ety_instr = '"etymology": "Brief root/origin",' if need_ety else ""
    
    prompt = f"""
I need to create Anki cards for the following English words. 
Output STRICT JSON Array.

**Word List:**
{words_str}

**Requirements:**
1. Fields per item (use exact keys):
   - "front_phrase": {phrase_instr}
   - "word": "The target word"
   - "meaning": Definition in {lang_instr}.
   - "example_sentence": {ex_count} sentence(s).
   {ety_instr}
   
**JSON Example:**
[
  {{ "word": "abandon", "front_phrase": "abandon ship", "meaning": "放弃", "example_sentence": "He decided to abandon the plan." }}
]
"""
    return prompt.strip()

# ==========================================
# 5. Anki 生成逻辑 (已修复 KeyError)
# ==========================================
def parse_anki_data(json_input):
    """
    解析 JSON，具备极强的容错能力。
    1. 自动处理 'w', 'm' 等简写键名。
    2. 自动处理逐行 JSON (JSON Lines)。
    3. 自动忽略错误行。
    """
    raw_items = []
    
    # --- 阶段 1: 尝试解析 JSON 结构 ---
    try:
        # 尝试匹配整个数组 [...]
        match = re.search(r'\[.*\]', json_input, re.DOTALL)
        if match:
            raw_items = json.loads(match.group(0))
        else:
            # 如果不是数组，尝试按行解析
            lines = json_input.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                if line.endswith(','): line = line[:-1] # 去掉行尾逗号
                try:
                    raw_items.append(json.loads(line))
                except:
                    continue
    except:
        return []

    if not isinstance(raw_items, list):
        # 如果解析出来不是列表（比如是单个对象），包一层
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        else:
            return []

    # --- 阶段 2: 标准化键名 (Mapping) ---
    # 无论 AI 返回什么怪异的 Key，这里统一转换成标准 Key
    standardized_data = []
    
    for item in raw_items:
        if not isinstance(item, dict): continue
        
        # 提取数据，优先找全称，找不到找缩写，再找不到给空字符串
        # 1. 正面/短语
        front = item.get('front_phrase') or item.get('w') or item.get('word') or "Unknown"
        
        # 2. 单词本身
        word = item.get('word') or item.get('w') or front
        
        # 3. 释义
        meaning = item.get('meaning') or item.get('m') or item.get('def') or ""
        
        # 4. 例句 (注意：你的 Traceback 里用的是 'examples'，AI 给的是 'e' 或 'example_sentence')
        example = item.get('example_sentence') or item.get('examples') or item.get('e') or ""
        
        # 5. 词源
        ety = item.get('etymology') or item.get('r') or item.get('root') or ""

        # 构造标准字典
        new_item = {
            'front_phrase': str(front),
            'word': str(word),
            'meaning': str(meaning),
            'examples': str(example), # 统一叫 examples
            'etymology': str(ety)
        }
        standardized_data.append(new_item)
            
    return standardized_data

def generate_anki_package(data, deck_name):
    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)
    
    # 这里的 CSS 引用 styles.ANKI_CSS
    my_model = genanki.Model(
        model_id,
        'Vocab Flow Ultra Model',
        fields=[
            {'name': 'FrontPhrase'},
            {'name': 'Word'},
            {'name': 'Meaning'},
            {'name': 'Etymology'},
            {'name': 'Example'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '<div class="phrase">{{FrontPhrase}}</div><div class="word">({{Word}})</div>',
                'afmt': '''
                {{FrontSide}}
                <hr id=answer>
                <div class="meaning">{{Meaning}}</div>
                <div class="example">{{Example}}</div>
                <div class="ety">{{Etymology}}</div>
                ''',
            },
        ],
        css=styles.ANKI_CSS
    )

    my_deck = genanki.Deck(deck_id, deck_name)

    for c in data:
        # 使用 .get() 方法，防止 KeyError
        # 并把 \n 换行符转换为 HTML 的 <br>
        ex_text = c.get('examples', '').replace('\n', '<br>')
        
        note = genanki.Note(
            model=my_model,
            fields=[
                c.get('front_phrase', ''), 
                c.get('word', ''),         
                c.get('meaning', ''),      
                c.get('etymology', ''),    
                ex_text
            ]
        )
        my_deck.add_note(note)

    output_path = f"{deck_name}.apkg"
    genanki.Package(my_deck).write_to_file(output_path)
    return output_path