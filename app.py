import streamlit as st
import pandas as pd
import re
import os

# ==========================================
# 1. 极简配置与样式
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master", page_icon="🅰️")

st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.5;
        font-family: 'Consolas', 'Courier New', monospace; /* 方便阅读的等宽字体 */
    }
    .stNumberInput input { font-weight: bold; color: #1a73e8; }
    .block-container { padding-top: 2rem; }
    /* 隐藏多余元素 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 内置智能还原引擎 (零依赖，纯 Python)
# ==========================================
# 手动维护的高频不规则词表 (覆盖 95% 常见场景)
IRREGULAR_MAP = {
    "is": "be", "am": "be", "are": "be", "was": "be", "were": "be", 
    "been": "be", "being": "be", "'s": "be", "'re": "be", "'m": "be",
    "has": "have", "had": "have", "having": "have", "'ve": "have",
    "does": "do", "did": "do", "done": "do", "doing": "do",
    "went": "go", "gone": "go", "going": "go", "goes": "go",
    "made": "make", "making": "make", "makes": "make",
    "took": "take", "taken": "take", "taking": "take",
    "came": "come", "coming": "come", "comes": "come",
    "saw": "see", "seen": "see", "seeing": "see",
    "knew": "know", "known": "know", "knowing": "know",
    "got": "get", "gotten": "get", "getting": "get",
    "gave": "give", "given": "give", "giving": "give",
    "told": "tell", "telling": "tell",
    "felt": "feel", "feeling": "feel",
    "became": "become", "becoming": "become",
    "left": "leave", "leaving": "leave",
    "put": "put", "putting": "put",
    "meant": "mean", "meaning": "mean",
    "kept": "keep", "keeping": "keep",
    "let": "let", "letting": "let",
    "began": "begin", "begun": "begin", "beginning": "begin",
    "seemed": "seem", "seeming": "seem",
    "helped": "help", "helping": "help",
    "showed": "show", "shown": "show", "showing": "show",
    "heard": "hear", "hearing": "hear",
    "played": "play", "playing": "play",
    "ran": "run", "running": "run",
    "moved": "move", "moving": "move",
    "lived": "live", "living": "live",
    "believed": "believe", "believing": "believe",
    "brought": "bring", "bringing": "bring",
    "happened": "happen", "happening": "happen",
    "wrote": "write", "written": "write", "writing": "write",
    "provided": "provide", "providing": "provide",
    "sat": "sit", "sitting": "sit",
    "stood": "stand", "standing": "stand",
    "lost": "lose", "losing": "lose",
    "paid": "pay", "paying": "pay",
    "met": "meet", "meeting": "meet",
    "included": "include", "including": "include",
    "continued": "continue", "continuing": "continue",
    "set": "set", "setting": "set",
    "learnt": "learn", "learned": "learn", "learning": "learn",
    "changed": "change", "changing": "change",
    "led": "lead", "leading": "lead",
    "understood": "understand", "understanding": "understand",
    "watched": "watch", "watching": "watch",
    "followed": "follow", "following": "follow",
    "stopped": "stop", "stopping": "stop",
    "created": "create", "creating": "create",
    "spoke": "speak", "spoken": "speak", "speaking": "speak",
    "read": "read", "reading": "read",
    "allowed": "allow", "allowing": "allow",
    "added": "add", "adding": "add",
    "spent": "spend", "spending": "spend",
    "grew": "grow", "grown": "grow", "growing": "grow",
    "opened": "open", "opening": "open",
    "walked": "walk", "walking": "walk",
    "won": "win", "winning": "win",
    "offered": "offer", "offering": "offer",
    "remembered": "remember", "remembering": "remember",
    "loved": "love", "loving": "love",
    "considered": "consider", "considering": "consider",
    "appeared": "appear", "appearing": "appear",
    "bought": "buy", "buying": "buy",
    "waited": "wait", "waiting": "wait",
    "served": "serve", "serving": "serve",
    "died": "die", "dying": "die",
    "sent": "send", "sending": "send",
    "expected": "expect", "expecting": "expect",
    "built": "build", "building": "build",
    "stayed": "stay", "staying": "stay",
    "fell": "fall", "fallen": "fall", "falling": "fall",
    "cut": "cut", "cutting": "cut",
    "reached": "reach", "reaching": "reach",
    "killed": "kill", "killing": "kill",
    "remained": "remain", "remaining": "remain",
    "better": "good", "best": "good",
    "worse": "bad", "worst": "bad",
    "mice": "mouse", "feet": "foot", "teeth": "tooth",
    "children": "child", "men": "man", "women": "woman"
}

def get_smart_lemma(word, vocab_set):
    """
    智能还原逻辑：
    1. 查不规则表 (went -> go)
    2. 查词库 (如果词库里有 families，直接认)
    3. 规则去尾 (families -> family, liked -> like)
    """
    # 1. 已经在词库里 (比如 'the')
    if word in vocab_set: return word
    
    # 2. 查不规则表
    if word in IRREGULAR_MAP: return IRREGULAR_MAP[word]
    
    # 3. 规则去尾尝试
    # 尝试去掉 's (users' -> user)
    if word.endswith("'s") or word.endswith("’s"):
        base = word[:-2]
        if base in vocab_set: return base
        
    # 尝试 ies -> y (families -> family)
    if word.endswith("ies"):
        base = word[:-3] + "y"
        if base in vocab_set: return base
        
    # 尝试 es -> "" (boxes -> box)
    if word.endswith("es"):
        base = word[:-2]
        if base in vocab_set: return base
        
    # 尝试 s -> "" (cats -> cat)
    if word.endswith("s") and not word.endswith("ss"):
        base = word[:-1]
        if base in vocab_set: return base

    # 尝试 ed -> "" (liked -> like) 或 ed -> e (lived -> live)
    if word.endswith("ed"):
        base1 = word[:-2] # played -> play
        if base1 in vocab_set: return base1
        base2 = word[:-1] # lived -> live
        if base2 in vocab_set: return base2

    # 尝试 ing -> "" 或 ing -> e
    if word.endswith("ing"):
        base1 = word[:-3] # going -> go
        if base1 in vocab_set: return base1
        base2 = word[:-3] + "e" # making -> make
        if base2 in vocab_set: return base2

    return word # 实在还原不了，返回原词

# ==========================================
# 3. 词库加载 (coca_cleaned.csv)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if not file_path: return None
    
    try:
        df = pd.read_csv(file_path)
        # 极简清洗：只认 word 和 rank 列，忽略大小写
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        
        # 智能匹配列名
        w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
        r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
        
        # 建立高效查词字典
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
        
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
    except:
        return None

vocab_dict = load_vocab()

# ==========================================
# 4. 界面布局 (Google Translate 风格)
# ==========================================

# 顶部设置栏
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    current_level = st.number_input("当前水平 (Current Level)", 0, 20000, 6000, 500)
with c2:
    target_level = st.number_input("目标水平 (Target Level)", 0, 20000, 8000, 500)

st.divider()

# 左右分栏
left, right = st.columns([1, 1])

with left:
    st.caption("输入文本 (Input Text)")
    text_input = st.text_area("input_area", height=500, placeholder="在此粘贴英语文章...", label_visibility="collapsed")
    analyze_btn = st.button("⚡ 开始分析 / Analyze", type="primary", use_container_width=True)

with right:
    st.caption("分析结果 (Analysis Result)")
    
    if not vocab_dict:
        st.error("❌ 未找到词库文件 (coca_cleaned.csv)")
    elif analyze_btn and text_input:
        
        # 1. 文本预处理 (正则分词，只留字母)
        # 这一步自动过滤了中文、标点、数字
        words = re.findall(r'[a-z]+', text_input.lower())
        unique_words = sorted(list(set(words)))
        
        data = []
        vocab_keys = set(vocab_dict.keys()) # 加速查找
        
        # 2. 查词逻辑
        for w in unique_words:
            if len(w) < 2: continue # 跳过单个字母
            
            # 智能还原
            lemma = get_smart_lemma(w, vocab_keys)
            
            # 查排名
            rank = vocab_dict.get(lemma, 99999)
            
            # 分组逻辑
            category = "beyond"
            if rank <= current_level:
                category = "known"
            elif rank <= target_level:
                category = "target"
            
            data.append({
                "word": lemma,
                "rank": rank,
                "category": category
            })
            
        # 3. 生成结果
        df = pd.DataFrame(data)
        
        if not df.empty:
            df = df.sort_values('rank')
            
            t1, t2, t3 = st.tabs([
                f"🟡 重点词 ({len(df[df['category']=='target'])})", 
                f"🔴 超纲词 ({len(df[df['category']=='beyond'])})", 
                f"🟢 已掌握 ({len(df[df['category']=='known'])})"
            ])
            
            # 渲染纯文本列表的函数
            def show_list(category_name):
                subset = df[df['category'] == category_name]
                if subset.empty:
                    st.info("列表为空")
                else:
                    # 拼接成文本：abandon (6023)
                    text_content = "\n".join([f"{row['word']} ({row['rank']})" for _, row in subset.iterrows()])
                    st.text_area(f"{category_name}_out", value=text_content, height=450, label_visibility="collapsed")

            with t1: show_list("target")
            with t2: show_list("beyond")
            with t3: show_list("known")
            
        else:
            st.warning("未检测到有效英文单词。")

    elif not text_input:
        st.info("👈 请在左侧输入文本")