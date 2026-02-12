# app.py
import streamlit as st
import pandas as pd
import time
import random

# 引入本地模块
import utils
import logic
import styles
from utils import VOCAB_DICT, FULL_DF

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 动态 Key 初始化 (用于一键清空)
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

st.markdown(styles.CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

with tab_guide:
    st.markdown("""
    ### 👋 欢迎使用 Vocab Flow Ultra
    这是一个**从阅读材料中提取生词**，并利用 **AI** 自动生成 **Anki 卡片**的效率工具。
    
    ---
    
    <div class="guide-step">
    <span class="guide-title">Step 1: 提取生词 (Extract)</span>
    在 <code>1️⃣ 单词提取</code> 标签页：<br><br>
    <strong>1. 上传文件</strong><br>
    支持 PDF, TXT, EPUB, DOCX。无论是小说、文章还是单词表，直接丢进去即可。<br>
    系统会自动进行 <strong>NLP 词形还原</strong>（将 went 还原为 go）并清洗垃圾词（乱码、重复字符）。<br>
    <br>
    <strong>2. 设置过滤范围 (Rank Filter)</strong><br>
    利用 COCA 20000 词频表进行科学筛选：
    <ul>
        <li><strong>忽略排名前 N</strong> (Min Rank)：例如设为 <code>2000</code>，会过滤掉 `the, is, you` 等最基础的高频词。</li>
        <li><strong>忽略排名后 N</strong> (Max Rank)：例如设为 <code>15000</code>，会过滤掉极其生僻的词。</li>
        <li><strong>🔓 包含生僻词</strong> (Unknown)：勾选后，将强制包含词频表中没有的词（如人名、地名、新造词）。</li>
    </ul>
    <br>
    <strong>3. 点击 🚀 开始分析</strong><br>
    系统会融合处理，自动去重并按词频排序，最大化提取有效单词。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 2: 获取 Prompt (AI Generation)</span>
    分析完成后：<br><br>
    <strong>1. 自定义设置</strong><br>
    点击 <code>⚙️ 自定义 Prompt 设置</code>，选择正面是单词还是短语，释义语言等。<br>
    <br>
    <strong>2. 复制 Prompt</strong><br>
    系统会自动将单词分组。生成的单词表支持<strong>折叠</strong>和<strong>滚动查看</strong>。<br>
    <ul>
        <li>📱 <strong>手机/鸿蒙端</strong>：使用下方的“纯文本框”，长按全选 -> 复制。</li>
        <li>💻 <strong>电脑端</strong>：点击代码块右上角的 Copy 📄 图标。</li>
    </ul>
    <br>
    <strong>3. 发送给 AI</strong><br>
    将复制的内容发送给 ChatGPT / Claude / Gemini / DeepSeek。AI 会返回一串 JSON 数据。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 3: 制作 Anki 牌组 (Create Deck)</span>
    在 <code>2️⃣ Anki 制作</code> 标签页：<br><br>
    <strong>1. 粘贴 AI 回复</strong><br>
    将 AI 生成的 JSON 内容粘贴到输入框中。<br>
    <div class="guide-tip">💡 <strong>支持追加粘贴</strong>：如果你有 5 组单词，可以把 AI 的 5 次回复依次粘贴在同一个框里，不需要分批下载。</div>
    <br>
    <strong>2. 下载与导入</strong><br>
    点击 <strong>📥 下载 .apkg</strong>，然后双击该文件，它会自动导入到你的 Anki 软件中。
    </div>
    """, unsafe_allow_html=True)

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析", "🔢 词频列表"])
    
    with mode_context:
        # V29: 统一模式，只保留筛选器
        st.info("💡 **全能模式**：系统将自动进行 NLP 词形还原、去重、垃圾词清洗。无论是文章还是单词表，直接上传即可。")
        
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略排名前 N 的词", 1, 20000, 100, step=100)
        targ = c2.number_input("忽略排名后 N 的词", 2000, 50000, 20000, step=500)
        include_unknown = st.checkbox("🔓 包含生僻词/人名 (Rank > 20000)", value=False)

        uploaded_file = st.file_uploader("📂 上传文档 (TXT/PDF/DOCX/EPUB)", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("📄 ...或粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 读取文件并清洗垃圾词...")
                raw_text = logic.extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 2:
                    status.write("🔍 智能分析与词频比对...")
                    
                    # 统一调用，不再区分模式
                    final_data, raw_count = logic.analyze_logic(raw_text, curr, targ, include_unknown)
                    
                    st.session_state['gen_words_data'] = final_data # [(word, rank), ...]
                    st.session_state['raw_count'] = raw_count
                    st.session_state['process_time'] = time.time() - start_time
                    
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容太短", state="error")
        
        if st.button("🗑️ 清空", type="secondary", on_click=utils.clear_all_state): pass

    with mode_rank:
        gen_type = st.radio("模式", ["🔢 顺序", "🔀 随机"], horizontal=True)
        if "顺序" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
             count = c_b.number_input("数量", 10, 500, 50, step=10)
             if st.button("🚀 生成"):
                 start_time = time.time()
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                     # 构造统一格式 [(word, rank), ...]
                     data_list = list(zip(subset[w_col], subset[r_col]))
                     st.session_state['gen_words_data'] = data_list
                     st.session_state['raw_count'] = 0
                     st.session_state['process_time'] = time.time() - start_time
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("Min Rank", 1, 20000, 1, step=100)
             max_r = c_max.number_input("Max Rank", 1, 25000, 5000, step=100)
             r_count = c_cnt.number_input("Count", 10, 200, 50, step=10)
             if st.button("🎲 抽取"):
                 start_time = time.time()
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                     candidates = FULL_DF[mask]
                     if len(candidates) > 0:
                         subset = candidates.sample(n=min(r_count, len(candidates))).sort_values(r_col)
                         data_list = list(zip(subset[w_col], subset[r_col]))
                         st.session_state['gen_words_data'] = data_list
                         st.session_state['raw_count'] = 0
                         st.session_state['process_time'] = time.time() - start_time

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        # 解包数据
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        st.markdown("### 📊 分析报告")
        k1, k2, k3 = st.columns(3)
        raw_c = st.session_state.get('raw_count', 0)
        p_time = st.session_state.get('process_time', 0.1)
        k1.metric("📄 文档总字数", f"{raw_c:,}")
        k2.metric("🎯 筛选生词 (已去重)", f"{len(words_only)}")
        k3.metric("⚡ 耗时", f"{p_time:.2f}s")
        
        # --- V29: 增强版预览区 (折叠+Rank) ---
        show_rank = st.checkbox("显示单词 Rank", value=False)
        
        # 构造显示文本
        if show_rank:
            display_text = ", ".join([f"{w}[{r}]" for w, r in data_pairs])
        else:
            display_text = ", ".join(words_only)
            
        with st.expander("📋 **全部生词预览 (点击展开/折叠)**", expanded=False):
            # 使用自定义 CSS 实现滚动容器
            st.markdown(f'<div class="scrollable-text">{display_text}</div>', unsafe_allow_html=True)
            st.caption("提示：长按上方文本框可全选复制，或点击下方代码块复制按钮。")
            st.code(display_text, language="text")

        with st.expander("⚙️ **自定义 Prompt 设置 (点击展开)**", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("正面内容", ["短语搭配 (Phrase)", "单词 (Word)"])
            def_mode = col_s2.selectbox("背面释义", ["英文", "中文", "中英双语"])
            
            col_s3, col_s4 = st.columns(2)
            ex_count = col_s3.slider("例句数量", 1, 3, 1)
            need_ety = col_s4.checkbox("包含词源/词根", value=True)

        batch_size = st.number_input("AI 分组大小", 10, 200, 100, step=10)
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 第 {idx+1} 组 (共 {len(batch)} 词)", expanded=(idx==0)):
                prompt_text = logic.get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.caption("📱 手机端专用：")
                st.text_area(f"text_area_{idx}", value=prompt_text, height=100, label_visibility="collapsed")
                st.caption("💻 电脑端：")
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 📦 制作 Anki")
    bj_time_str = utils.get_beijing_time_str()
    if 'anki_input_text' not in st.session_state: st.session_state['anki_input_text'] = ""

    st.caption("👇 粘贴 AI 回复：")
    ai_resp = st.text_area("JSON 输入框", height=300, key="anki_input_text")
    deck_name = st.text_input("牌组名", f"Vocab_{bj_time_str}")
    
    if ai_resp.strip():
        parsed_data = logic.parse_anki_data(ai_resp)
        if parsed_data:
            st.success(f"✅ 成功解析 {len(parsed_data)} 条数据")
            df_view = pd.DataFrame(parsed_data)
            df_view.rename(columns={'front_phrase': '正面', 'meaning': '背面', 'etymology': '词源'}, inplace=True)
            st.dataframe(df_view[['正面', '背面', '词源']], use_container_width=True, hide_index=True)
            
            f_path = logic.generate_anki_package(parsed_data, deck_name)
            with open(f_path, "rb") as f:
                st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
        else:
            st.warning("⚠️ 等待粘贴...")