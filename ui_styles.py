# UI styles for Vocab Flow Ultra. Loaded once in app.py.

APP_STYLES_HTML = """
<style>
    /* ===== Global: hide Streamlit chrome, set base font ===== */
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
                     'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-size-adjust: 100%;
        -webkit-text-size-adjust: 100%;
        font-size: 19px;
    }

    /* iOS safe-area support + 放宽主内容区宽度以更好利用空间 */
    .main .block-container {
        padding-left: max(1rem, env(safe-area-inset-left));
        padding-right: max(1rem, env(safe-area-inset-right));
        padding-bottom: max(1rem, env(safe-area-inset-bottom));
        max-width: 1400px;
    }

    /* ===== Smooth transitions on all interactive elements ===== */
    button, input, textarea, [data-baseweb="tab"], .stExpander {
        transition: all 0.2s ease !important;
    }

    /* ===== Buttons: pill-shaped, elevated feel ===== */
    .stButton>button {
        border-radius: 10px; font-weight: 600; width: 100%; margin-top: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        letter-spacing: 0.01em;
        min-height: 48px;
        font-size: 18px;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }
    .stButton>button:active {
        transform: scale(0.93);
        box-shadow: 0 0 0 rgba(0,0,0,0);
        filter: brightness(0.88);
        transition: transform 0.06s ease, box-shadow 0.06s ease, filter 0.06s ease !important;
    }

    /* Form submit buttons (e.g. lookup) – stronger active feedback */
    .stForm [data-testid="stFormSubmitButton"] button {
        position: relative;
        overflow: hidden;
    }
    .stForm [data-testid="stFormSubmitButton"] button:active {
        transform: scale(0.90);
        box-shadow: 0 0 0 rgba(0,0,0,0);
        filter: brightness(0.85);
        transition: transform 0.06s ease, filter 0.06s ease !important;
    }

    /* Ripple animation for all buttons */
    @keyframes btn-ripple {
        0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.35); }
        100% { box-shadow: 0 0 0 12px rgba(59,130,246,0); }
    }
    .stButton>button:focus-visible,
    .stForm [data-testid="stFormSubmitButton"] button:focus-visible {
        animation: btn-ripple 0.4s ease-out;
    }

    /* ===== Text areas ===== */
    .stTextArea textarea {
        font-family: 'Consolas', 'SF Mono', 'Monaco', monospace;
        font-size: 18px; border-radius: 10px;
        -webkit-overflow-scrolling: touch;
    }

    /* ===== Form cards (only for bordered forms) ===== */
    .stForm:not([class*="borderless"]) {
        border-radius: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    /* ===== Text input: comfortable height ===== */
    .stTextInput input {
        min-height: 50px;
        font-size: 19px;
    }

    /* ===== Metric cards ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1rem; border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700; letter-spacing: -0.02em; font-size: 1.6rem;
    }

    /* ===== Tabs: segmented-control style ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #f1f5f9; padding: 4px;
        border-radius: 12px; border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.1rem; border-radius: 10px;
        font-weight: 600; font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* ===== Expanders ===== */
    .stExpander {
        border: 1px solid #e5e7eb; border-radius: 12px;
        margin-bottom: 10px; overflow: hidden;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* ===== Progress bar ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 6px;
    }

    /* ===== Section dividers ===== */
    hr { border: none; height: 1px; background: #e5e7eb; margin: 1.5rem 0; }

    /* ===== AI lookup result styles ===== */
    .ql-result { padding: 4px 0; }
    .ql-word   { color: #1e3a8a; margin-bottom: 6px; font-size: 18px; line-height: 1.7; }
    .ql-def    { color: #1e3a8a; margin-bottom: 6px; font-size: 18px; line-height: 1.7; }
    .ql-etym   { color: #065f46; background: #ecfdf5; padding: 6px 10px; border-radius: 8px; margin: 8px 0; line-height: 1.7; }
    .ql-ex     { color: #374151; margin-top: 6px; font-size: 18px; line-height: 1.7; }
    .ql-misc   { color: #6b7280; margin-bottom: 8px; font-size: 18px; line-height: 1.7; }
    .ql-stream { padding: 10px 12px; border: 1px solid #dbeafe; background: #f8fbff; border-radius: 10px; color: #1f2937; line-height: 1.7; font-size: 18px; }

    /* ===== App footer ===== */
    .app-footer {
        margin-top: 3rem; padding: 1.25rem 0; text-align: center;
        color: #94a3b8; font-size: 0.8rem; letter-spacing: 0.02em;
        border-top: 1px solid #f1f5f9;
    }

    /* ===== Hero header ===== */
    .app-hero {
        text-align: center; padding: 1.5rem 0 0.5rem;
    }
    .app-hero h1 {
        font-size: 2.45rem; font-weight: 900; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.2rem;
    }
    .app-hero p {
        color: #64748b; font-size: 1.12rem; margin: 0;
        font-weight: 600;
    }
    /* ===== Radio buttons: chip style ===== */
    .stRadio > div { gap: 0.4rem; display: grid !important; grid-template-columns: repeat(2, 1fr) !important; }
    .stRadio > div > label {
        border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 0.35rem 0.85rem; font-size: 0.95rem;
        transition: all 0.15s ease;
    }
    .stRadio > div > label:hover {
        border-color: #93c5fd; background: #f0f9ff;
    }
    /* ② 生成方式：内置 AI / 第三方 AI 选项更大 */
    #ai-gen-mode-radio ~ div .stRadio label {
        font-size: 1.28rem !important;
        padding: 0.6rem 1.2rem !important;
        min-height: 2.8rem !important;
    }
    /* 一键生成按钮更紧凑（位于 35% 窄列内） */
    #ai-gen-mode-radio ~ * [data-testid="column"] .stButton > button {
        min-height: 40px !important;
        font-size: 16px !important;
    }

    /* ===== Number inputs ===== */
    .stNumberInput input { border-radius: 10px; }
    .stTextInput input, .stSelectbox input, .stNumberInput input {
        min-height: 46px;
        font-size: 18px;
    }

    /* ===== Toast / info / warning boxes ===== */
    .stAlert { border-radius: 10px; }
    .stMarkdown p, .stCaption, label, .stRadio label, .stCheckbox label {
        font-size: 1.08rem;
    }

    /* ===== Download button ===== */
    .stDownloadButton > button {
        border-radius: 10px; font-weight: 600;
        box-shadow: 0 2px 8px rgba(59,130,246,0.15);
        min-height: 48px;
        font-size: 18px;
    }

    /* ===== Dark mode / night UI refinement ===== */
    @media (prefers-color-scheme: dark) {
        /* -- Base -- */
        .stApp {
            background: linear-gradient(180deg, #0b1120 0%, #0f172a 50%, #0b1120 100%);
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] {
            background: #0f172a !important;
            border-right: 1px solid #1e293b;
        }

        /* -- Hero gradient for dark bg -- */
        .app-hero h1 {
            background: linear-gradient(135deg, #60a5fa 0%, #818cf8 50%, #a78bfa 100%);
            -webkit-background-clip: text; background-clip: text;
        }
        .app-hero p { color: #94a3b8; }

        /* -- Forms & cards -- */
        .stForm {
            background: #111827;
            border-color: #1e293b;
            box-shadow: 0 1px 4px rgba(0,0,0,0.45);
        }

        /* -- Inputs: text, number, select, textarea -- */
        .stTextInput input, .stNumberInput input, .stSelectbox input {
            background: #1e293b; color: #e2e8f0;
            border-color: #334155;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox input:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 2px rgba(96,165,250,0.25);
        }
        .stTextInput input::placeholder { color: #64748b; }
        .stTextArea textarea {
            background: #1e293b; color: #e2e8f0;
            border-color: #334155;
        }
        .stTextArea textarea:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 2px rgba(96,165,250,0.25);
        }
        .stTextArea textarea::placeholder { color: #64748b; }

        /* -- Buttons -- */
        .stButton>button {
            border-color: #334155;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        }
        .stDownloadButton > button {
            box-shadow: 0 2px 8px rgba(96,165,250,0.2);
        }

        /* -- Metric cards -- */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-color: #1e293b;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9; }
        [data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #94a3b8; }

        /* -- Tabs -- */
        .stTabs [data-baseweb="tab-list"] {
            background: #0f172a;
            border-color: #1e293b;
        }
        .stTabs [data-baseweb="tab"] { color: #94a3b8; }
        .stTabs [data-baseweb="tab"]:hover { color: #cbd5e1; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #1e293b;
            color: #f1f5f9;
            box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        }

        /* -- Expanders -- */
        .stExpander {
            border-color: #1e293b;
            background: #111827;
            box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        .stExpander summary span { color: #e2e8f0; }

        /* -- Alerts: semi-transparent bg lets Streamlit's semantic tint + border colours show -- */
        .stAlert {
            background: rgba(15, 23, 42, 0.88);
            color: #e2e8f0;
        }
        [data-testid="stAlert"][data-baseweb="notification"] {
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }

        /* -- Radio chips -- */
        .stRadio > div > label {
            border-color: #334155;
            color: #cbd5e1;
            background: #1e293b;
        }
        .stRadio > div > label:hover {
            border-color: #60a5fa;
            background: #172554;
            color: #e2e8f0;
        }

        /* -- Checkbox -- */
        .stCheckbox label { color: #cbd5e1; }

        /* -- Progress bar dark glow -- */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #3b82f6 0%, #818cf8 100%);
        }

        /* -- Markdown text -- */
        .stMarkdown p, .stMarkdown li, .stMarkdown h1,
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #e2e8f0; }
        .stCaption { color: #94a3b8 !important; }
        label { color: #cbd5e1; }

        /* -- Code block -- */
        .stCodeBlock, pre, code {
            background: #0f172a !important;
            color: #e2e8f0 !important;
            border-color: #1e293b !important;
        }

        /* -- Dataframe -- */
        .stDataFrame { border-color: #1e293b; }

        /* -- Dividers & footer -- */
        hr { background: #1e293b; }
        .app-footer { color: #64748b; border-top-color: #1e293b; }

        /* -- File uploader -- */
        [data-testid="stFileUploader"] {
            border-color: #334155;
        }
        [data-testid="stFileUploader"] section {
            background: #1e293b;
            border-color: #334155;
        }

        /* -- Spinner text -- */
        .stSpinner > div { color: #94a3b8; }

        /* -- AI lookup result card -- */
        .ql-result { background: #1e293b !important; border-color: #334155 !important; }
        .ql-word { color: #93c5fd !important; }
        .ql-def  { color: #c4b5fd !important; }
        .ql-etym { background: #1a2332 !important; color: #6ee7b7 !important; }
        .ql-ex   { color: #cbd5e1 !important; }
        .ql-misc { color: #94a3b8 !important; }

        /* -- Streaming box -- */
        .ql-stream { background: #1e293b !important; border-color: #334155 !important; color: #e2e8f0 !important; }

        /* -- Scrollbar -- */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }

        /* -- Link color -- */
        a { color: #60a5fa; }
        a:hover { color: #93c5fd; }

        /* -- SelectBox / Multiselect trigger area -- */
        [data-baseweb="select"] > div:first-child {
            background: #1e293b !important;
            border-color: #334155 !important;
            color: #e2e8f0 !important;
        }

        /* -- Dropdown options panel (BaseWeb portal, outside stApp) -- */
        [data-baseweb="popover"] {
            background: #1e293b !important;
            border-color: #334155 !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
        }
        [data-baseweb="menu"] { background: #1e293b !important; }
        li[role="option"] {
            background: #1e293b !important;
            color: #e2e8f0 !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background: #1e3a5f !important;
            color: #f1f5f9 !important;
        }

        /* -- MultiSelect tags -- */
        [data-baseweb="tag"] {
            background: #172554 !important;
            color: #93c5fd !important;
            border-color: #1e3a5f !important;
        }

        /* -- Number input stepper buttons (+/−) -- */
        .stNumberInput button {
            background: #1e293b !important;
            border-color: #334155 !important;
            color: #94a3b8 !important;
        }
        .stNumberInput button:hover {
            background: #334155 !important;
            color: #e2e8f0 !important;
        }

        /* -- Tooltip icon & popover -- */
        [data-testid="stTooltipIcon"] svg { fill: #64748b !important; }
        [data-testid="stTooltipHover"] {
            background: #1e293b !important;
            border-color: #334155 !important;
            color: #e2e8f0 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
        }

        /* -- st.status() widget -- */
        [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"] > div {
            background: #1e293b !important;
            border-color: #334155 !important;
            color: #e2e8f0 !important;
        }
    }

    /* ===== Mobile (iOS/Android) ===== */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.8rem;
            padding-left: max(0.75rem, env(safe-area-inset-left));
            padding-right: max(0.75rem, env(safe-area-inset-right));
            padding-bottom: max(1rem, env(safe-area-inset-bottom));
        }
        .app-hero {
            padding: 0.9rem 0 0.2rem;
        }
        .app-hero h1 {
            font-size: 1.95rem;
        }
        .app-hero p {
            font-size: 0.92rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.55rem 0.75rem;
            font-size: 0.9rem;
            min-height: 40px;
        }
        .stButton>button,
        .stDownloadButton > button {
            min-height: 46px;
            border-radius: 12px;
        }
        /* Force lookup form columns to stay side-by-side on mobile */
        .stForm [data-testid="stColumns"] {
            flex-wrap: nowrap !important;
            gap: 6px !important;
        }
        .stForm [data-testid="stFormSubmitButton"] button {
            min-width: 72px;
            min-height: 46px;
            font-size: 14px;
            padding: 0 8px;
        }
        .stForm .stTextInput input {
            min-height: 46px;
        }
        /* Avoid hover lift on touch devices */
        .stButton>button:hover {
            transform: none;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        /* Stronger touch feedback on mobile */
        .stButton>button:active,
        .stForm [data-testid="stFormSubmitButton"] button:active {
            transform: scale(0.90) !important;
            filter: brightness(0.82) !important;
            transition: transform 0.05s ease, filter 0.05s ease !important;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .stCaption {
            font-size: 0.9rem;
        }
    }
</style>
"""
