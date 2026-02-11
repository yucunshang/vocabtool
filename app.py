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
import pysrt
# API 库
from openai import OpenAI

# ==========================================
# 1. 核心配置与 NLTK 修复
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

# 修复 Streamlit Cloud 上的 NLTK 报错 (增加 punkt_tab)
@st.cache_resource
def download_nltk_data():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'), # 关键修复
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_data()

# ==========================================
# 2. 工具函数定义
# ==========================================

def get_exam_tag(rank):
    """根据 COCA 排名映射考试难度"""
    if pd.isna(rank): return "未知"
    rank = int(rank)
    if rank <= 2000: return "初中/基础"
    if rank <= 4000: return "高中/四级"
    if rank <= 6000: return "六级/考研"
    if rank <= 9000: return "雅思/托福"
    if rank <= 13000: return "GRE/专八"
    if rank <= 20000: return "高阶原著"
    return "超纲/罕见"

def extract_text_from_file(uploaded_file):
    """多格式文本提取器"""
    text = ""
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'txt':
            text = uploaded_file.getvalue().decode("utf-8")
            
        elif file_ext == 'pdf':
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
                
        elif file_ext == 'epub':
            # EbookLib 需要文件路径，先存临时文件
            with open("temp.epub", "wb") as f:
                f.write(uploaded_file.getbuffer())
            book = epub.read_epub("temp.epub")
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text += soup.get_text() + "\n"
            os.remove("temp.epub") # 清理
            
        elif file_ext in ['srt', 'vtt']:
            content = uploaded_file.getvalue().decode("utf-8")
            # 简单正则去除时间轴 (00:00:01,000 --> ...)
            lines = [l for l in content.splitlines() 
                     if not re.match(r'(\d{2}:\d{2})|(\d+$)', l.strip())]
            text = "\n".join(lines)
            
    except Exception as e:
        st.error(f"文件解析失败: {str(e)}")
        return ""
        
    return text

def smart_lemmatize(text):
    """智能词形还原 (带词性判断)"""
    # 清洗非字母字符，但保留空格
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = nltk.word_tokenize(clean_text)
    pos_tags = nltk.pos_tag(words)
    
    lemmatized = []
    for word, tag in pos_tags:
        if len(word) < 2: continue # 跳过单个字母
        
        # 映射 Treebank POS 到 lemminflect
        if tag.startswith('J'): pos = 'ADJ'
        elif tag.startswith('V'): pos = 'VERB'
        elif tag.startswith('R'): pos = 'ADV'
        else: pos = 'NOUN'
        
        lemma = lemminflect.getLemma(word, upos=pos)
        if not lemma: 
            lemma = word.lower()
        else: 
            lemma = lemma[0].lower()
            
        lemmatized.append(lemma)
    return lemmatized

# ==========================================
# 3. 侧边栏设置
# ==========================================
with st.sidebar:
    st.title("⚙️ 设置")
    st.markdown("### DeepSeek API")
    api_key = st.text_input("API Key", type="password", placeholder="sk-...", help="填入 Key 可直接生成解释，否则仅复制 Prompt")
    
    st.markdown("### 筛选配置")
    rank_range = st.slider("词频范围 (Rank)", 0, 20000, (4000, 15000), help="数字越大单词越生僻")
    
    st.markdown("---")
    st.info("💡 提示：支持上传 PDF, EPUB, SRT 字幕文件")

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.title("🚀 Vocab Master Pro")
st.caption("上传文档 -> 智能提取生词 -> 一键生成 Anki 卡片")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("📂 上传文件", type=['txt', 'pdf', 'epub', 'srt'])
with col2:
    user_input = st.text_area("✍️ 或直接粘贴文本", height=100)

# 获取文本
raw_text = ""
if uploaded_file:
    raw_text = extract_text_from_file(uploaded_file)
elif user_input:
    raw_text = user_input

if raw_text:
    # --- 处理开始 ---
    with st.spinner("正在进行 NLP 分析与词频比对..."):
        start_time = time.time()
        
        # 1. 还原
        words = smart_lemmatize(raw_text)
        
        # 2. 统计
        word_counts = pd.Series(words).value_counts().reset_index()
        word_counts.columns = ['word', 'count']
        
        # 3. 加载 COCA 数据 (带容错)
        try:
            # 尝试加载真实数据
            df_coca = pd.read_csv('coca20000.csv') 
            # 简单的列名标准化，防止csv表头不一样
            if 'lemma' in df_coca.columns: 
                df_coca.rename(columns={'lemma': 'word'}, inplace=True)
        except FileNotFoundError:
            #如果没有文件，生成模拟数据防止报错
            st.toast("⚠️ 未找到 coca20000.csv，使用测试数据运行", icon="🐞")
            import numpy as np
            mock_words = word_counts['word'].tolist()
            df_coca = pd.DataFrame({
                'word': mock_words,
                'rank': np.random.randint(1, 20000, size=len(mock_words))
            })
            
        # 4. 合并数据
        df_merged = pd.merge(word_counts, df_coca, on='word', how='inner') # inner join 只保留认识的词
        
        # 5. 增加标签
        df_merged['Exam_Tag'] = df_merged['rank'].apply(get_exam_tag)
        
        # 6. 筛选
        mask = (df_merged['rank'] >= rank_range[0]) & (df_merged['rank'] <= rank_range[1])
        df_final = df_merged[mask].sort_values('rank').reset_index(drop=True)
        
        end_time = time.time()

    # --- 结果展示 ---
    st.divider()
    st.success(f"✅ 分析完成！耗时 {end_time - start_time:.2f}s | 原文 {len(words)} 词 | 命中生词 **{len(df_final)}** 个")

    if not df_final.empty:
        # 可视化区域
        tab1, tab2 = st.tabs(["📊 难度分布", "📈 词频趋势"])
        with tab1:
            st.bar_chart(df_final['Exam_Tag'].value_counts())
        with tab2:
            st.line_chart(df_final['rank'])
            
        # 数据表区域
        st.subheader("📝 生词列表")
        st.dataframe(
            df_final[['word', 'rank', 'Exam_Tag', 'count']], 
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("词频排名 (COCA)"),
                "count": st.column_config.ProgressColumn("出现次数", format="%d", min_value=0, max_value=df_final['count'].max())
            }
        )
        
        # --- AI 生成区域 ---
        st.divider()
        st.subheader("🤖 AI 解释生成")
        
        # 构建 Prompt
        target_words = df_final['word'].head(50).tolist() # 限制前50个防止 token 溢出
        words_str = ", ".join(target_words)
        
        default_prompt = f"""
你是一个专业的英语老师。请分析以下单词，并输出为 CSV 格式（使用竖线 | 分隔）。
包含字段：单词 | 音标 | 词性 | 中文简明释义 | 语境例句 (英文) | 记忆法 (词根/联想)

单词列表：
{words_str}

要求：
1. 不要输出表头。
2. 释义要精准简练。
3. 严格遵守格式，方便导入 Anki。
"""
        user_prompt = st.text_area("Prompt 预览 (可编辑)", value=default_prompt, height=200)
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            # 简单的复制按钮逻辑（Streamlit原生不支持点击复制，这里用代码块展示方便复制）
            st.info("👇 这里的 Prompt 已准备好，全选复制即可")
            st.code(user_prompt, language="text")
            
        with c2:
            st.write("### 🚀 直接调用 DeepSeek")
            if st.button("开始生成 (DeepSeek V3)", type="primary"):
                if not api_key:
                    st.error("请先在左侧边栏填入 API Key")
                else:
                    try:
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                        
                        placeholder = st.empty()
                        full_response = ""
                        
                        # 流式输出效果
                        with st.spinner("AI 正在思考中..."):
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "你是一个辅助生成 Anki 卡片的助手。"},
                                    {"role": "user", "content": user_prompt}
                                ],
                                stream=True
                            )
                            
                            for chunk in response:
                                if chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    full_response += content
                                    placeholder.markdown(full_response + "▌")
                                    
                            placeholder.markdown(full_response)
                            st.success("生成完毕！您可以直接复制上方内容存为 .csv 文件导入 Anki。")
                            
                    except Exception as e:
                        st.error(f"API 调用失败: {e}")

    else:
        st.warning("在此筛选条件下没有找到生词，请尝试调整左侧 Rank 范围。")