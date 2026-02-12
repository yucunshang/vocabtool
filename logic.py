# logic.py
import re
import os
import json
import random
import utils
from utils import VOCAB_DICT, load_nlp_resources, get_file_parsers, get_genanki
from styles import ANKI_CSS

# ==========================================
# 核心逻辑 (V29: 融合算法)
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            for encoding in ['utf-8', 'gb18030', 'latin-1']:
                try:
                    text = bytes_data.decode(encoding)
                    break
                except: continue
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'epub':
            genanki, tempfile = get_genanki()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator=' ', strip=True) + " "
            os.remove(tmp_path)
    except Exception as e:
        return f"Error: {e}"
    return text

def is_valid_word(word):
    """
    垃圾词清洗
    """
    if len(word) < 2: return False
    if len(word) > 25: return False 
    # 连续3个相同字符 -> 认为是垃圾词
    if re.search(r'(.)\1{2,}', word): return False
    # 没有元音 -> 认为是缩写或乱码 (排除 hmm, brrr, zszs)
    if not re.search(r'[aeiouy]', word): return False
    return True

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    """
    V29 核心算法：混合增强匹配
    同时检查 [单词原形] 和 [Lemma 还原词]，只要任意一个在词频表且符合 Rank 范围，即命中。
    """
    nltk, lemminflect = load_nlp_resources()
    
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    # 1. 宽松分词 (保留 internal hyphens)
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(raw_tokens)
    
    # 2. 清洗 + 小写 + 初步去重
    clean_tokens = set([t.lower() for t in raw_tokens if is_valid_word(t.lower())])
    
    final_candidates = [] # 存储 (display_word, rank)
    seen_lemmas = set()   # 用于去重逻辑，防止 go 和 went 同时出现
    
    for w in clean_tokens:
        # A. 计算 Lemma
        lemma = get_lemma_local(w)
        
        # B. 获取 Rank (优先查 Lemma，查不到查原词)
        # 很多词频表里只有 go，没有 went。但也有少数情况原词有排名。
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        # 取最靠前的有效排名 (非99999的最小值)
        if rank_lemma != 99999 and rank_orig != 99999:
            best_rank = min(rank_lemma, rank_orig)
        elif rank_lemma != 99999:
            best_rank = rank_lemma
        else:
            best_rank = rank_orig
            
        # C. 判定是否符合范围
        is_in_range = (best_rank >= current_lvl and best_rank <= target_lvl)
        is_unknown_included = (best_rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            # D. 去重核心逻辑
            # 我们希望输出的是 Lemma (例如输出 go 而不是 went)，这样对背单词更友好
            # 但如果 Lemma 是未知词，而原词是已知词(极少见)，则保留原词
            
            word_to_keep = lemma if rank_lemma != 99999 else w
            
            # 使用 lemma 作为去重键值 (Key)
            # 这样 went(go) 和 go(go) 会被视为同一个，只保留一个
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    # E. 排序：Rank 小的在前 (高频 -> 低频)，未知词(99999)放最后
    final_candidates.sort(key=lambda x: x[1])
    
    return final_candidates, total_words

def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.replace("```json", "").replace("```", "").strip()
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases_lower = set()

    for match in matches:
        json_str = match.group()
        try:
            data = json.loads(json_str, strict=False)
            front_text = data.get("w", "").strip()
            meaning = data.get("m", "").strip()
            examples = data.get("e", "").strip()
            etymology = data.get("r", "").strip()
            
            if not etymology or etymology.lower() == "none" or etymology == "":
                etymology = ""

            if not front_text or not meaning: continue
            
            front_text = front_text.replace('**', '')
            
            if front_text.lower() in seen_phrases_lower: 
                continue
            seen_phrases_lower.add(front_text.lower())

            parsed_cards.append({
                'front_phrase': front_text,
                'meaning': meaning,
                'examples': examples,
                'etymology': etymology
            })
        except: continue
    return parsed_cards

# ==========================================
# Anki 生成
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
    
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id, f'VocabFlow JSON Model {model_id}',
        fields=[{'name': 'FrontPhrase'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{FrontPhrase}}</div>', 
            'afmt': '''
            {{FrontSide}}<hr>
            <div class="definition">{{Meaning}}</div>
            <div class="examples">{{Examples}}</div>
            {{#Etymology}}
            <div class="footer-info"><div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div></div>
            {{/Etymology}}
            ''',
        }], css=ANKI_CSS
    )
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[str(c['front_phrase']), str(c['meaning']), str(c['examples']).replace('\n','<br>'), str(c['etymology'])]))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# Prompt Logic
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    if front_mode == "单词 (Word)":
        w_instr = "Key `w`: The word itself (lowercase)."
    else:
        w_instr = "Key `w`: A short practical collocation/phrase (2-5 words)."

    if def_mode == "中文":
        m_instr = "Key `m`: Concise Chinese definition (max 10 chars)."
    elif def_mode == "中英双语":
        m_instr = "Key `m`: English Definition + Chinese Definition."
    else:
        m_instr = "Key `m`: English definition (concise)."

    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."

    if need_ety:
        r_instr = "Key `r`: Simplified Chinese Etymology (Root/Prefix)."
    else:
        r_instr = "Key `r`: Leave this empty string \"\"."

    return f"""
Task: Create Anki cards.
Words: {w_list}

**OUTPUT: NDJSON (One line per object).**

**Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Keys:** `w` (Front), `m` (Meaning), `e` (Examples), `r` (Etymology)

**Example:**
{{"w": "...", "m": "...", "e": "...", "r": "..."}}

**Start:**
"""