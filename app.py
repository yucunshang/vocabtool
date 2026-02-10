import streamlit as st
import pandas as pd
import re
import os
import simplemma

# ==========================================
# 0. 核心修复：强制内置映射表 (不再依赖下载)
# ==========================================
# 这里手动定义最常见的不规则动词，确保 100% 能还原
# 即使 simplemma 挂了，这些词也能对！
MANUAL_LEMMAS = {
    "is": "be", "am": "be", "are": "be", "was": "be", "were": "be", 
    "been": "be", "being": "be", "'s": "be", "'re": "be", "'m": "be",
    "has": "have", "had": "have", "having": "have", "'ve": "have",
    "does": "do", "did": "do", "done": "do", "doing": "do",
    "went": "go", "gone": "go", "going": "go", "goes": "go",
    "made": "make", "making": "make", "makes": "make",
    "took": "take", "taken": "take", "taking": "take",
    "came": "come", "coming": "come",
    "saw": "see", "seen": "see",
    "knew": "know", "known": "know",
    "got": "get", "gotten": "get",
    "gave": "give", "given": "give",
    "told": "tell",
    "felt": "feel",
    "became": "become",
    "left": "leave",
    "put": "put",
    "meant": "mean",
    "kept": "keep",
    "let": "let",
    "began": "begin", "begun": "begin",
    "seemed": "seem",
    "helped": "help",
    "showed": "show",
    "heard": "hear",
    "played": "play",
    "ran": "run",
    "moved": "move",
    "lived": "live",
    "believed": "believe",
    "brought": "bring",
    "happened": "happen",
    "wrote": "write", "written": "write",
    "provided": "provide",
    "sat": "sit",
    "stood": "stand",
    "lost": "lose",
    "paid": "pay",
    "met": "meet",
    "included": "include",
    "continued": "continue",
    "set": "set",
    "learnt": "learn", "learned": "learn",
    "changed": "change",
    "led": "lead",
    "understood": "understand",
    "watched": "watch",
    "followed": "follow",
    "stopped": "stop",
    "created": "create",
    "spoke": "speak", "spoken": "speak",
    "read": "read",
    "allowed": "allow",
    "added": "add",
    "spent": "spend",
    "grew": "grow",
    "opened": "open",
    "walked": "walk",
    "won": "win",
    "offered": "offer",
    "remembered": "remember",
    "loved": "love",
    "considered": "consider",
    "appeared": "appear",
    "bought": "buy",
    "waited": "wait",
    "served": "serve",
    "died": "die",
    "sent": "send",
    "expected": "expect",
    "built": "build",
    "stayed": "stay",
    "fell": "fall", "fallen": "fall",
    "cut": "cut",
    "reached": "reach",
    "killed": "kill",
    "remained": "remain"
}

def get_lemma_robust(word):
    """三保险还原策略"""
    # 1. 第一层保险：查手动表 (处理最高频的不规则词)
    if word in MANUAL_LEMMAS:
        return MANUAL_LEMMAS[word]
    
    # 2. 第二层保险：Simplemma (尝试调用)
    try:
        res = simplemma.lemmatize(word, lang='en')
        if res != word: return res
    except:
        pass
        
    # 3. 第三层保险：简单规则去尾 (处理规则复数/动词)
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ly'):
        return word[:-2]
        
    return word

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="Vibe Vocab Studio", page_icon="⚡", layout="wide")
st.title("⚡ Vibe Vocab v9.0 (硬核还原版)")
st.caption("内置高频变形表 · 专治 'are/been' 不认识")

# ==========================================
# 2. 读取词库
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab_simple():
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
            
    if not file_path: return None, "未找到文件"

    # 优先读 coca_cleaned
    if 'cleaned' in file_path:
        try:
            df = pd.read_csv(file_path)
            # 确保列名正确
            if 'word' in df.columns and 'rank' in df.columns:
                vocab = pd.Series(df['rank'].values, index=df['word'].astype(str)).to_dict()
                return vocab, "加载成功 (Cleaned)"
        except: pass

    # 兜底读原始文件
    for enc in ['utf-8', 'utf-8-sig', 'gbk']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            # 找列
            cols = [str(c).lower() for c in df.columns]
            df.columns = cols
            
            w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c or '排序' in c or '词频' in c), cols[1] if len(cols)>1 else cols[0])
            
            # 清洗
            df['w'] = df[w_col].astype(str).str.lower().str.strip()
            df['r'] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
            
            vocab = pd.Series(df['r'].values, index=df['w']).to_dict()
            return vocab, "加载成功 (Raw)"
        except: continue
        
    return None, "加载失败"

vocab_dict, msg = load_vocab_simple()

if not vocab_dict:
    st.error(msg)
    st.stop()
    
# 侧边栏自检
st.sidebar.success(f"📚 {msg}")
check_are = vocab_dict.get('be', 'Not Found')
st.sidebar.info(f"检查点: 'be' 排名 = {check_are}")
st.sidebar.info(f"还原测试: went -> {get_lemma_robust('went')}")

# ==========================================
# 3. 核心逻辑 (调用强力还原)
# ==========================================
st.sidebar.divider()
vocab_range = st.sidebar.slider("学习区间", 1, 20000, (6000, 8000), 500)
r_start, r_end = vocab_range

def process_text(text):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z\']{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    known, target, beyond = [], [], []
    
    for w in unique_words:
        rank = 99999
        match = w
        note = ""

        # A. 直接查 (is -> is?)
        if w in vocab_dict:
            rank = vocab_dict[w]
        
        # B. 强力还原查 (is -> be)
        if rank > 20000: # 如果直接查没查到，或者查到了但排名很低(可能是错误条目)
            lemma = get_lemma_robust(w)
            if lemma in vocab_dict:
                # 只有当还原后的排名更靠前时，才采纳
                lemma_rank = vocab_dict[lemma]
                if lemma_rank < rank:
                    rank = lemma_rank
                    match = lemma
                    note = f"<{w}>"

        item = {'单词': match, '排名': int(rank), '备注': note}
        
        if rank <= r_start: known.append(item)
        elif r_start < rank <= r_end: target.append(item)
        else: beyond.append(item)

    return pd.DataFrame(known), pd.DataFrame(target), pd.DataFrame(beyond)

# ==========================================
# 4. 界面
# ==========================================
text_input = st.text_area("在此粘贴文本:", height=150)

if st.button("🚀 开始分析", type="primary"):
    if not text_input: st.warning("请输入内容")
    else:
        df_k, df_t, df_b = process_text(text_input)
        
        st.success("分析完成")
        t1, t2, t3 = st.tabs([
            f"🟡 重点 ({len(df_t)})", 
            f"🔴 生词/超纲 ({len(df_b)})", 
            f"🟢 熟词 ({len(df_k)})"
        ])
        
        with t1: st.dataframe(df_t, use_container_width=True)
        with t2: st.dataframe(df_b, use_container_width=True)
        with t3: st.dataframe(df_k, use_container_width=True)