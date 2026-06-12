import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from src.preprocess import preprocess_text

# ==============================================================================
# 1. STRUCTURAL PAGE CONFIGURATION & STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="AI Bug Severity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize deep session state persistent logs to prevent single-input resetting
if "bug_data" not in st.session_state:
    st.session_state.bug_data = pd.DataFrame(columns=["Description", "Predicted_Severity", "Solution"])

# Load machine learning assets safely
try:
    model = joblib.load("models/severity_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except Exception:
    st.error(
        "⚠️ Model assets missing. Please run `src/train.py` first to compile your models."
    )

# ==============================================================================
# 2. PIXEL-PERFECT HIGH-FIDELITY PREMIUM DARK UI OVERRIDES (CSS Injection)
# ==============================================================================
st.markdown(
    """
    <style>
        /* Base Deep Viewport Color Palette Match */
        .stApp {
            background: radial-gradient(circle at top right, #131e31, #0a0f18) !important;
            color: #cbd5e1 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Dark Sidebar customization match */
        [data-testid="stSidebar"] {
            background-color: #0b111c !important;
            border-right: 1px solid #1e293b;
        }
        
        /* Top Navigation Title Header styling */
        .header-title-container {
            font-size: 22px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 0.8px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Glassmorphic Metric Box Containers */
        .glass-metric-row {
            display: flex;
            gap: 14px;
            margin-bottom: 25px;
            width: 100%;
        }
        
        .glass-metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6)) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            flex: 1;
            position: relative;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(5px);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-text-block {
            display: flex;
            flex-direction: column;
        }
        
        .metric-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8 !important;
            margin-bottom: 4px;
        }
        
        .metric-digits {
            font-size: 32px;
            font-weight: 700;
            color: #ffffff !important;
            line-height: 1;
        }
        
        /* Bottom glowing border markers matching mockup highlights */
        .glow-blocker { border-bottom: 3px solid #ef4444 !important; }
        .glow-critical { border-bottom: 3px solid #f97316 !important; }
        .glow-major { border-bottom: 3px solid #facc15 !important; }
        .glow-minor { border-bottom: 3px solid #3b82f6 !important; }
        .glow-trivial { border-bottom: 3px solid #10b981 !important; }
        
        /* Layout Framework Clean Headers */
        .section-panel-title {
            font-size: 16px;
            font-weight: 500;
            color: #ffffff;
            margin-top: 10px;
            margin-bottom: 15px;
            letter-spacing: 0.3px;
        }

        /* Native Streamlit Textarea Input element adjustments to match dark look */
        div[data-testid="stTextArea"] textarea {
            background-color: #0d1522 !important;
            border: 1px solid #1e293b !important;
            color: #f1f5f9 !important;
            border-radius: 6px !important;
            font-size: 14px;
        }

        /* Native Streamlit Buttons Override to Premium Navy/Blue Glow */
        div[data-testid="stButton"] button {
            background: linear-gradient(180deg, #2563eb, #1d4ed8) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }

        /* Custom styling override rules for high fidelity premium Dataframes */
        .stDataFrame div {
            border-radius: 6px;
        }
        
        /* Clean default gaps layout framework fix */
        div[data-testid="stBlock"] {
            padding-top: 0rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. HELPER TRANSFORMATION & DICTIONARY ENGINES
# ==============================================================================
def get_solution_meta(sev):
    solutions = {
        "blocker": "Fatal crash in UI login module... Check memory allocations.",
        "critical": "DB connection fails under high load... Validate connection pooling.",
        "major": "Functional search results incomplete... Optimize index rules.",
        "normal": "Standard use case failure in cart workflow validation steps.",
        "minor": "Medium low functional boundary adjustment exception logged.",
        "trivial": "Text overlap on profile page dashboard asset... Fix CSS alignment.",
    }
    return solutions.get(sev.lower(), "Review logging metrics for trace debugging.")

# ==============================================================================
# 4. SIDEBAR NAVIGATION CONTROLS
# ==============================================================================
st.sidebar.markdown(
    "<h2 style='color:#ffffff; font-size:18px; font-weight:600; margin-bottom:2px; letter-spacing:0.5px;'>AI BUG SEVERITY ANALYZER</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='color:#475569; font-size:12px; margin-top:0; margin-bottom:25px;'>⚙️ System Navigation</p>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:2px;'>Radio menu</p>", unsafe_allow_html=True)
menu = st.sidebar.radio("", ["🏠 Home", "🔄 Continue"], label_visibility="collapsed")

st.sidebar.markdown("<br><p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:5px;'>Select CSV with 'Description' row</p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["csv"], label_visibility="collapsed")

# Handle bulk batch CSV modifications immediately within state memory
if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    if "Description" in df_upload.columns and "processed_file" not in st.session_state:
        df_upload["clean"] = df_upload["Description"].apply(preprocess_text)
        X_batch = vectorizer.transform(df_upload["clean"])
        df_upload["Predicted_Severity"] = model.predict(X_batch)
        df_upload["Predicted_Severity"] = df_upload["Predicted_Severity"].str.upper().str.strip()
        df_upload["Solution"] = df_upload["Predicted_Severity"].apply(get_solution_meta)
        
        # Merge batch rows smoothly with current runtime database list
        st.session_state.bug_data = pd.concat(
            [st.session_state.bug_data, df_upload[["Description", "Predicted_Severity", "Solution"]]], 
            ignore_index=True
        )
        st.session_state.processed_file = True

export_placeholder = st.sidebar.empty()

# Persistent state safe clean download pipeline
if not st.session_state.bug_data.empty:
    csv_bytes = st.session_state.bug_data.to_csv(index=False).encode("utf-8")
    export_placeholder.download_button(
        label="📥 Export Priority Sorted Sheet",
        data=csv_bytes,
        file_name="priority_sorted_bugs.csv",
        mime="text/csv",
    )

st.sidebar.markdown("<br><br><br><br><br><hr style='border-color:#1e293b;'>", unsafe_allow_html=True)
st.sidebar.caption("System Version 1.7.0")

# ==============================================================================
# 5. HEADER TITLE & INPUT CONTROLLER ACTIONS
# ==============================================================================
st.markdown("<div class='header-title-container'>🐞 AI BUG SEVERITY ANALYZER</div>", unsafe_allow_html=True)

mid_left_panel, mid_right_panel = st.columns([1, 2], gap="large")

with mid_left_panel:
    st.markdown("<div class='section-panel-title'>Single Input</div>", unsafe_allow_html=True)
    with st.container():
        st.button("Run Diagnostics", key="trigger_diagnostics_system")
        manual_input_entry = st.text_area(
            "Describe the bug...",
            height=125,
            placeholder="Describe the bug...",
            label_visibility="collapsed",
            key="single_bug_textarea"
        )
        
        # INTERACTIVE DISPATCH LOOP: Parses single textbox variables instantly inside session state
        if st.button("Classify and Solve", key="classify_and_solve_system"):
            if manual_input_entry.strip():
                clean_text = preprocess_text(manual_input_entry)
                X_single = vectorizer.transform([clean_text])
                predicted_label = model.predict(X_single)[0].upper().strip()
                calculated_solution = get_solution_meta(predicted_label)
                
                new_row = pd.DataFrame([{
                    "Description": manual_input_entry,
                    "Predicted_Severity": predicted_label,
                    "Solution": calculated_solution
                }])
                
                st.session_state.bug_data = pd.concat([st.session_state.bug_data, new_row], ignore_index=True)

# ==============================================================================
# 6. DYNAMIC COMPUTATION MATRIX (KPI ENGINE)
# ==============================================================================
log_dataframe = st.session_state.bug_data.copy()
if not log_dataframe.empty:
    log_dataframe["Predicted_Severity"] = log_dataframe["Predicted_Severity"].str.upper()

counts = log_dataframe["Predicted_Severity"].value_counts() if not log_dataframe.empty else {}

b_val = counts.get("BLOCKER", 0)
c_val = counts.get("CRITICAL", 0)
m_val = counts.get("MAJOR", 0) + counts.get("NORMAL", 0)
mi_val = counts.get("MINOR", 0)
t_val = counts.get("TRIVIAL", 0)

# ==============================================================================
# 7. GENERATING SPARKLINES FOR HIGH FIDELITY TOP METRIC CARDS
# ==============================================================================
def make_sparkline_svg(color, seed_val):
    np.random.seed(seed_val)
    points = np.random.randint(10, 35, size=8)
    path_data = "M " + " L ".join([f"{i*14},{p}" for i, p in enumerate(points)])
    return f"""<svg width="100" height="40" style="opacity: 0.7; overflow: visible;"><path d="{path_data}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

# ==============================================================================
# 8. ROW 1 LAYOUT: DYNAMIC HTML TOP KPI METRICS BLOCK
# ==============================================================================
html_metrics_string = f"""
<div class="glass-metric-row">
    <div class="glass-metric-card glow-blocker">
        <div class="metric-text-block">
            <span class="metric-label">BLOCKER</span>
            <span class="metric-digits">{b_val}</span>
        </div>
        {make_sparkline_svg('#ef4444', 42)}
    </div>
    <div class="glass-metric-card glow-critical">
        <div class="metric-text-block">
            <span class="metric-label">CRITICAL</span>
            <span class="metric-digits">{c_val}</span>
        </div>
        {make_sparkline_svg('#f97316', 15)}
    </div>
    <div class="glass-metric-card glow-major">
        <div class="metric-text-block">
            <span class="metric-label">MAJOR</span>
            <span class="metric-digits">{m_val}</span>
        </div>
        {make_sparkline_svg('#facc15', 88)}
    </div>
    <div class="glass-metric-card glow-minor">
        <div class="metric-text-block">
            <span class="metric-label">MINOR</span>
            <span class="metric-digits">{mi_val}</span>
        </div>
        {make_sparkline_svg('#3b82f6', 23)}
    </div>
    <div class="glass-metric-card glow-trivial">
        <div class="metric-text-block">
            <span class="metric-label">TRIVIAL</span>
            <span class="metric-digits">{t_val}</span>
        </div>
        {make_sparkline_svg('#10b981', 61)}
    </div>
</div>
"""

st.markdown(html_metrics_string, unsafe_allow_html=True)

# ==============================================================================
# 9. ROW 2 LAYOUT: SEVERITY BREAKDOWN BAR GRAPH PIPELINE
# ==============================================================================
with mid_right_panel:
    st.markdown("<div class='section-panel-title'>Severity Breakdown</div>", unsafe_allow_html=True)
    
    categories = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
    chart_values = [b_val, c_val, m_val, mi_val, t_val]
    premium_bar_colors = ['#ef4444', '#f97316', '#facc15', '#3b82f6', '#10b981']
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=chart_values,
        marker_color=premium_bar_colors,
        width=0.45,
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=185,
        xaxis=dict(tickfont=dict(color='#64748b', size=11), showgrid=False, fixedrange=True),
        yaxis=dict(tickfont=dict(color='#475569', size=10), gridcolor='#1e293b', showgrid=True, fixedrange=True),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==============================================================================
# 10. ROW 3 LAYOUT: PREMIUM FULL ROW COLOR-WASHED HIGH-FIDELITY BUG LOG
# ==============================================================================
# ==============================================================================
# 10. PREMIUM FULL ROW COLOR-WASHED HIGH-FIDELITY BUG LOG (CORRECTED BADGES)
# ==============================================================================
# ==============================================================================
# 10. PREMIUM BUG LOG WITH CORRESPONDING SOLID BADGES (UPDATED)
# ==============================================================================
# ==============================================================================
# 10. COMBINED HIGH-FIDELITY BUG LOG: ROW SHADING WITH TEXT-TARGETED PILL BADGES
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY COMBINED BUG LOG: ROW SHADING WITH BRIGHT TEXT BADGES
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING, PILL BADGES & NATIVE TABLE SEARCH
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING, PILL BADGES & NATIVE TABLE SEARCH
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING, ACTIVE PILL BADGES & NATIVE SEARCH
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING, PILL BADGES & REACTIVE TABLE SEARCH
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING, NATIVE SEARCH & PILL SEVERITY BADGES
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING & PRECISE SEVERITY PILL HIGHLIGHTS
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING & COMPACT SEVERITY PILL HIGHLIGHTS
# ==============================================================================
# ==============================================================================
# 10. HIGH-FIDELITY BUG LOG: ROW SHADING & PRECISE SEVERITY PILL HIGHLIGHTS
# ==============================================================================
st.markdown("<br><div class='section-panel-title'>Sorted Bug Log</div>", unsafe_allow_html=True)

# Define hierarchy sorting priority levels
severity_order = {"BLOCKER": 1, "CRITICAL": 2, "MAJOR": 3, "NORMAL": 4, "MINOR": 5, "TRIVIAL": 6}
if not log_dataframe.empty:
    log_dataframe["rank_index"] = log_dataframe["Predicted_Severity"].map(severity_order).fillna(7)
    display_df = log_dataframe.sort_values(by="rank_index")[["Description", "Predicted_Severity", "Solution"]].copy()
else:
    display_df = pd.DataFrame(columns=["Description", "Predicted_Severity", "Solution"])

# 1. Row styling function: Handles the soft background color for the entire row
def style_bug_log_matrix(df):
    # Create an empty DataFrame with the exact same shape for styling strings
    styles_df = pd.DataFrame("", index=df.index, columns=df.columns)
    
    for idx, row in df.iterrows():
        severity = str(row["Predicted_Severity"]).upper().strip()
        
        # Default row background shading styles
        row_colors = {
            "BLOCKER": "background-color: #2a1215; color: #fecaca;",
            "CRITICAL": "background-color: #2c1a10; color: #fed7aa;",
            "MAJOR": "background-color: #272410; color: #fde68a;",
            "NORMAL": "background-color: #0f172a; color: #bfdbfe;",
            "MINOR": "background-color: #0f172a; color: #bfdbfe;",
            "TRIVIAL": "background-color: #0d1f14; color: #bbf7d0;"
        }
        base_style = row_colors.get(severity, "background-color: #0b111c; color: #cbd5e1;")
        
        # Apply the soft background color to all columns as a baseline
        styles_df.loc[idx, :] = base_style
        
        # Overwrite ONLY the 'Predicted_Severity' cell with a vibrant solid pill badge background
        badge_colors = {
            "BLOCKER": "background-color: #e11d48; color: #ffffff; font-weight: bold; border-radius: 8px; text-align: center;",
            "CRITICAL": "background-color: #ea580c; color: #ffffff; font-weight: bold; border-radius: 8px; text-align: center;",
            "MAJOR": "background-color: #eab308; color: #1e293b; font-weight: bold; border-radius: 8px; text-align: center;",
            "NORMAL": "background-color: #2563eb; color: #ffffff; font-weight: bold; border-radius: 8px; text-align: center;",
            "MINOR": "background-color: #3b82f6; color: #ffffff; font-weight: bold; border-radius: 8px; text-align: center;",
            "TRIVIAL": "background-color: #16a34a; color: #ffffff; font-weight: bold; border-radius: 8px; text-align: center;"
        }
        styles_df.loc[idx, "Predicted_Severity"] = badge_colors.get(severity, base_style)
        
    return styles_df

if not display_df.empty:
    # Clean text to guarantee CSS maps precisely
    display_df["Predicted_Severity"] = display_df["Predicted_Severity"].fillna("MINOR").str.upper().str.strip()

    # 2. Build the styled matrix structure via Pandas Styler
    styled_table_grid = (
        display_df.style
        .apply(style_bug_log_matrix, axis=None)
        .set_properties(**{
            'border-color': '#1e293b',
            'padding': '12px 16px'
        })
    )
    
    # 3. Render using clean, native st.dataframe text columns to preserve searching features
    st.dataframe(
        styled_table_grid,
        use_container_width=True,
        height=290,
        column_config={
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Predicted_Severity": st.column_config.TextColumn("Predicted_Severity", width="medium"),
            "Solution": st.column_config.TextColumn("Solution", width="large")
        },
        hide_index=True
    )
else:
    st.dataframe(display_df, use_container_width=True, height=280)