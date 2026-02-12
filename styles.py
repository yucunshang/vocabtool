# styles.py

# 核心 CSS：使用 Streamlit 系统变量 (var) 自动适配日/夜模式
CUSTOM_CSS = """
<style>
    /* 1. 全局输入框字体优化 */
    .stTextArea textarea { 
        font-family: 'Consolas', monospace; 
        font-size: 14px; 
    }
    
    /* 2. 按钮样式 */
    .stButton>button { 
        border-radius: 8px; 
        font-weight: 600; 
        width: 100%; 
        margin-top: 5px; 
    }
    
    /* 3. 统计框 */
    .stat-box { 
        padding: 15px; 
        background-color: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-radius: 8px; 
        text-align: center; 
        color: var(--text-color);
        margin-bottom: 20px; 
    }
    
    /* 隐藏多余菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 4. 滚动容器 (预览单词用) */
    .scrollable-text {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid var(--secondary-background-color);
        border-radius: 5px;
        background-color: var(--secondary-background-color); /* 自动跟随背景 */
        color: var(--text-color);                            /* 自动反色 */
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    /* 5. 指南步骤卡片 (核心修复) */
    .guide-step { 
        background-color: var(--secondary-background-color); 
        color: var(--text-color);
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    }
    
    .guide-title { 
        font-size: 18px; 
        font-weight: bold; 
        color: var(--text-color);
        margin-bottom: 10px; 
        display: block; 
    }
    
    .guide-tip { 
        font-size: 14px; 
        color: var(--text-color); 
        background: transparent; 
        border: 1px dashed var(--text-color);
        padding: 8px; 
        border-radius: 4px; 
        margin-top: 8px; 
        opacity: 0.8;
    }
</style>
"""

# Anki 卡片样式 (用于生成的 .apkg 文件)
ANKI_CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; line-height: 1.3; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 18px; line-height: 1.5; margin-bottom: 15px; text-align: left; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; line-height: 1.4; border: 1px solid #fef3c7; }
"""