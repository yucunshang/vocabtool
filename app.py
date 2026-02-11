import streamlit as st
import pandas as pd
import re
import lemminflect
import nltk
# 新增库
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import io

# 确保 NLTK 数据已下载
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# ================= 配置区 =================
COCA_FILE_PATH = 'coca20000.csv'  # 假设你已有这个文件
DEFAULT_PROMPT_TEMPLATE = """
你是一个专业的语言学习助手。请分析以下英语单词列表。
请为每个单词提供以下内容，并严格以 CSV 格式输出 (分隔符为 | )：
单词 | 音标 | 词性 | 中文简明释义 | 常见搭配(英文短语) | 记忆法(词根/联想)

单词列表：
{words}

注意：
1. 不要包含表头。
2. 释义要精简，适合背诵。
3. 如果单词是多义词，优先提供最常用的含义。
"""

# ================= 核心功能函数 =================

# 1. 映射逻辑 (Q1)
def get_exam_tag(rank):
    if pd.isna(rank): return "未知"
    rank = int(rank)
    if rank <= 2000: return "初中/基础"
    if rank <= 4000: return "高中/四级"
    if rank <= 6000: return "六级/考研"
    if rank <= 9000: return "雅思/托福"
    if rank <= 15000: return "GRE/专八"
    return "原著/超纲"

# 2. 多格式解析 (Q4)
def extract_text_from_file(uploaded_file):
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8")
            
        elif file_type == 'pdf':
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        elif file_type == 'epub':
            # 需要保存临时文件因为 ebooklib 不支持直接读 stream
            with open("temp.epub", "wb") as f:
                f.write(uploaded_file.getbuffer())
            book = epub.read_epub("temp.epub")
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text += soup.get_text() + "\n"
                    
        elif file_type in ['srt', 'vtt']:
            # 简单粗暴去时间轴
            content = uploaded_file.getvalue().decode("utf-8")
            # 移除时间轴和序号 (简单正则)
            lines = [l for l in content.splitlines() if not re.match(r'(\d{2}:\d{2})|(\d+$)', l)]
            text = "\n".join(lines)
            
    except Exception as e:
        st.error(f"解析文件失败: {str(e)}")
        
    return text

# 3. 优化版词形还原 (Q3) - 增加简单的词性判断
def smart_lemmatize(text):
    words = nltk.word_tokenize(text)
    # 获取上下文词性，帮助还原 (better -> good)
    pos_tags = nltk.pos_tag(words) 
    
    lemmatized_words = []
    for word, tag in pos_tags:
        if not word[0].isalpha(): continue # 过滤标点
        
        # 将 NLTK tag 转换为 lemminflect tag
        if tag.startswith('J'): tag_type = 'ADJ'
        elif tag.startswith('V'): tag_type = 'VERB'
        elif tag.startswith('R'): tag_type = 'ADV'
        else: tag_type = 'NOUN' # 默认
        
        lemma = lemminflect.getLemma(word, upos=tag_type)
        if not lemma: lemma = word.lower() # 兜底
        else: lemma = lemma[0] # getLemma返回列表
        
        lemmatized_words.append(lemma.lower())
        
    return lemmatized_words

# ================= UI 主程序 =================
st.title("🚀 Vocab Master Pro v2.0")

# 侧边栏：API 设置 (Q7)
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("DeepSeek API Key (可选)", type="password", help="填入后可直接一键生成解释")
    show_charts = st.checkbox("显示统计图表", value=True) # (Q5)

# 主区域
uploaded_file = st.file_uploader("上传文件 (支持 TXT, PDF, EPUB, SRT)", type=['txt', 'pdf', 'epub', 'srt'])
user_text = st.text_area("或直接粘贴文本", height=150)

if uploaded_file or user_text:
    # 1. 获取文本
    if uploaded_file:
        raw_text = extract_text_from_file(uploaded_file)
    else:
        raw_text = user_text
        
    if raw_text:
        # 2. 处理流程
        with st.spinner("正在智能解析 & 还原词形..."):
            # A. 还原
            words = smart_lemmatize(raw_text)
            
            # B. 频次统计
            word_counts = pd.Series(words).value_counts().reset_index()
            word_counts.columns = ['word', 'count']
            
            # C. 读取 COCA 数据库 (模拟)
            # 实际使用时请读取你的 coca20000.csv
            # df_coca = pd.read_csv(COCA_FILE_PATH) 
            # 这里做一个 Mock 数据方便你运行 demo
            mock_coca = pd.DataFrame({
                'word': ['the', 'apple', 'ephemeral', 'serendipity', 'abandon'],
                'rank': [1, 1000, 14000, 16000, 3000]
            })
            
            # D. 合并数据
            #df_merged = pd.merge(word_counts, df_coca, on='word', how='left')
            # 临时用 mock 演示
            df_merged = pd.merge(word_counts, mock_coca, on='word', how='left')
            
            # E. 增加考试标签 (Q1)
            df_merged['Exam_Tag'] = df_merged['rank'].apply(get_exam_tag)
            
            # F. 筛选逻辑 (这里假设筛选生词)
            # 用户可以交互式筛选 Rank 区间
            min_rank, max_rank = st.slider("选择词频范围 (Rank)", 0, 20000, (4000, 15000))
            filtered_df = df_merged[
                (df_merged['rank'] >= min_rank) & 
                (df_merged['rank'] <= max_rank)
            ].sort_values('rank')

        # 3. 结果展示区
        st.success(f"解析完成！原文共 {len(words)} 词，筛选出 {len(filtered_df)} 个目标生词。")
        
        # (Q5) 可视化反馈
        if show_charts and not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 难度分布 (考试标准)")
                exam_dist = filtered_df['Exam_Tag'].value_counts()
                st.bar_chart(exam_dist)
            with col2:
                st.markdown("### 词频分布")
                st.line_chart(filtered_df['rank'].reset_index(drop=True))

        # 4. 单词预览与 Prompt 生成
        st.subheader("📝 生词列表 & AI 生成")
        
        # 显示数据表
        st.dataframe(filtered_df[['word', 'count', 'rank', 'Exam_Tag']], use_container_width=True)
        
        # (Q6) 动态 Prompt
        target_words = filtered_df['word'].tolist()
        words_str = ", ".join(target_words[:50]) # 限制数量防止 Token 爆炸，实际可分批
        
        default_prompt = DEFAULT_PROMPT_TEMPLATE.format(words=words_str)
        
        st.markdown("### 🤖 发送给 AI")
        user_prompt = st.text_area("编辑 Prompt (可修改要求)", value=default_prompt, height=200)
        
        col_copy, col_run = st.columns([1, 1])
        with col_copy:
            st.code(user_prompt, language="text")
            st.caption("👆 点击右上角复制，去 ChatGPT/Claude 粘贴")
            
        with col_run:
            if st.button("🚀 使用 DeepSeek 直接生成 (需配置 Key)", type="primary"):
                if not api_key:
                    st.warning("请先在左侧边栏填入 API Key")
                else:
                    st.info("正在调用 DeepSeek V3 API (模拟)...")
                    # 这里接入 requests 调用 DeepSeek
                    # response = requests.post(...)
                    # st.markdown(response.choices[0].message.content)
                    st.success("API 调用功能需自行对接 requests 库实现，逻辑已通！")