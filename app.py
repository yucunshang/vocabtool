import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import time
# 文件处理库
import PyPDF2
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
# API 库
from openai import OpenAI

# ==========================================
# 0. 用户配置区 (在这里填你的 Key)
# ==========================================
# ⚠️⚠️⚠️ 请将你的 DeepSeek API Key 填在下面引号内 ⚠️⚠️⚠️
USER_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
API_BASE_URL = "https://api.deepseek.com"

# ==========================================
# 1. 基础环境设置
# ==========================================
st.set_page_config(layout="centered", page_title="Vocab Master", page_icon="⚡")

# NLTK 自动修复 (包含 punkt_tab)
@st.cache_resource
def init_nltk():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'wordnet']
    for res in resources:
        try:
            if 'punkt' in res: nltk.data.find(f'tokenizers/{res}')
            else: nltk.data.find(f'*/{res}')
        except LookupError:
            nltk.download(res)

init_nltk()

# ==========================================
# 2. 核心逻辑函数
# ==========================================
def get_exam_tag(rank):
    if pd.isna(rank): return "未知"
    rank = int(rank)
    if rank <= 2000: return "基础"
    if rank <= 4000: return "四级/高中"
    if rank <= 6000: return "六级/考研"
    if rank <= 9000: return "雅思/托福"
    if rank <= 13000: return "GRE/专八"
    return "高阶原著"

def extract_text(file):
    """通用文本提取"""
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'txt': return file.getvalue().decode("utf-8")
        if ext == 'pdf':
            pdf = PyPDF2.PdfReader(file)
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        if ext == 'epub':
            with open("temp.epub", "wb") as f: f.write(file.getbuffer())
            book = epub.read_epub("temp.epub")
            text = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    text += BeautifulSoup(item.get_body_content(), 'html.parser').get_text() + "\n"
            os.remove("temp.epub")
            return text
        if ext in ['srt', 'vtt']:
            lines = [l for l in file.getvalue().decode("utf-8").splitlines() 
                     if not re.match(r'(\d{2}:\d{2})|(\d+$)', l.strip())]
            return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
    return ""

def smart_lemmatize(text):
    clean = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = nltk.word_tokenize(clean)
    pos_tags = nltk.pos_tag(words)
    lemmas = []
    for w, t in pos_tags:
        if len(w) < 2: continue
        tag_map = {'J': 'ADJ', 'V': 'VERB', 'R': 'ADV'}
        pos = tag_map.get(t[0], 'NOUN')
        lemma = lemminflect.getLemma(w, upos=pos)
        lemmas.append(lemma[0].lower() if lemma else w.lower())
    return lemmas

# ==========================================
# 3. UI 界面 (仿原版，无侧边栏)
# ==========================================
st.title("⚡ Vocab Master")

# 第一行：上传 + 文本框 (紧凑布局)
c1, c2 = st.columns([1, 1])
with c1:
    f = st.file_uploader("导入文件 (TXT/PDF/EPUB/SRT)", type=['txt','pdf','epub','srt'])
with c2:
    t = st.text_area("直接粘贴文本", height=68, placeholder="在此粘贴英文内容...")

# 获取输入
raw = ""
if f: raw = extract_text(f)
elif t: raw = t

# 设置区 (折叠起来，不占地)
with st.expander("⚙️ 筛选设置 (Rank 范围 / 统计图表)", expanded=False):
    sc1, sc2 = st.columns(2)
    with sc1:
        # 使用原来的数字输入方式，可能比滑块更极客
        min_r = st.number_input("最小 Rank", value=4000, step=1000)
    with sc2:
        max_r = st.number_input("最大 Rank", value=15000, step=1000)
    show_chart = st.checkbox("显示可视化图表", value=True)

# 处理逻辑
if raw:
    st.divider()
    with st.spinner("Analyzing..."):
        # 1. 解析
        words = smart_lemmatize(raw)
        counts = pd.Series(words).value_counts().reset_index()
        counts.columns = ['word', 'count']
        
        # 2. 读取/生成数据
        try:
            coca = pd.read_csv('coca20000.csv')
            if 'lemma' in coca.columns: coca.rename(columns={'lemma':'word'}, inplace=True)
        except:
            # 没文件时的 Mock 数据
            import numpy as np
            coca = pd.DataFrame({'word': counts['word'], 'rank': np.random.randint(1,20000, len(counts))})

        # 3. 筛选
        merged = pd.merge(counts, coca, on='word')
        merged['tag'] = merged['rank'].apply(get_exam_tag)
        final = merged[(merged['rank']>=min_r) & (merged['rank']<=max_r)].sort_values('rank')

    # 结果展示
    st.markdown(f"**分析结果：** 原文 {len(words)} 词 | 🎯 命中生词 **{len(final)}** 个")
    
    if show_chart and not final.empty:
        st.bar_chart(final['tag'].value_counts())

    if not final.empty:
        # 紧凑的数据展示
        st.dataframe(
            final[['word', 'rank', 'tag', 'count']], 
            use_container_width=True,
            hide_index=True
        )
        
        # AI 生成区
        st.divider()
        st.subheader("DeepSeek 解释生成")
        
        # 自动提取前 50 个词
        target_list = final['word'].head(50).tolist()
        
        default_prompt = f"""请分析以下单词，输出 CSV 格式（竖线 | 分隔），不带表头。
字段：单词 | 音标 | 词性 | 中文简明释义 | 英文语境例句 | 记忆法
单词：{', '.join(target_list)}"""

        # 允许用户最后修改一下 Prompt
        final_prompt = st.text_area("Prompt", value=default_prompt, height=100)
        
        if st.button("🚀 开始生成 (使用内置 Key)", type="primary"):
            if "sk-" not in USER_API_KEY:
                st.error("请先在代码第 19 行填入正确的 API Key！")
            else:
                client = OpenAI(api_key=USER_API_KEY, base_url=API_BASE_URL)
                output_box = st.empty()
                full_text = ""
                
                try:
                    stream = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": final_prompt}],
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_text += chunk.choices[0].delta.content
                            output_box.code(full_text + "▌", language="csv")
                    
                    output_box.code(full_text, language="csv") # 最终结果去除光标
                    st.success("生成完成！可直接复制上方内容。")
                except Exception as e:
                    st.error(f"API 请求失败: {e}")