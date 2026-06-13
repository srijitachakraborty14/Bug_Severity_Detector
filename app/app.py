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
    page_icon="🪲",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "bug_data" not in st.session_state:
    st.session_state.bug_data = pd.DataFrame(columns=["Description", "Predicted_Severity", "Solution"])

try:
    model = joblib.load("models/severity_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except Exception:
    model = None
    vectorizer = None

# ==============================================================================
# 2. PREMIUM CSS OVERRIDES, GLOBAL VIEWPORT INJECTIONS & BUG CURSOR ENGINE
# ==============================================================================
st.markdown(
    """
    <style>
        /* ----------------------------------------------------------------------
           🕷️ CUSTOM BUG MOUSE CURSOR SYSTEM OVERRIDES
           Injects an optimized vector bug cursor across all interaction layers
        ---------------------------------------------------------------------- */
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size:24px;'><text y='24'>🪲</text></svg>"), auto !important;
        }
        
        /* Ensures interactive buttons, dropdowns, and text areas retain the custom cursor shape */
        button, select, textarea, input, label, a, .stDownloadButton, [data-testid="stMarkdownContainer"] p {
            cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size:24px;'><text y='24'>🪲</text></svg>"), pointer !important;
        }
        /* ---------------------------------------------------------------------- */

        /* Transparent application viewport frame layer to showcase the canvas movement underneath */
        .stApp {
            background: transparent !important;
            color: #cbd5e1 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Sidebar dark baseline styling alignment */
        [data-testid="stSidebar"] {
            background-color: rgba(7, 12, 20, 0.95) !important;
            border-right: 1px solid #141f32;
            backdrop-filter: blur(4px);
        }
        
        /* Top Navigation Title Header styling */
        .header-title-container {
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }

        /* Glassmorphic Metric Box Containers */
        .glass-metric-row {
            display: flex;
            gap: 14px;
            margin-bottom: 25px;
            width: 100%;
        }
        
        .glass-metric-card {
            background: linear-gradient(135deg, rgba(18, 30, 49, 0.75), rgba(10, 15, 26, 0.9)) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            flex: 1;
            position: relative;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(12px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        
        .glass-metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(255, 255, 255, 0.2) !important;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.8);
        }
        
        .metric-text-block {
            display: flex;
            flex-direction: column;
        }
        
        .metric-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #94a3b8 !important;
            margin-bottom: 4px;
        }
        
        .metric-digits {
            font-size: 34px;
            font-weight: 700;
            color: #ffffff !important;
            line-height: 1;
        }
        
        .glow-blocker { border-bottom: 3px solid #e11d48 !important; }
        .glow-critical { border-bottom: 3px solid #ea580c !important; }
        .glow-major { border-bottom: 3px solid #eab308 !important; }
        .glow-minor { border-bottom: 3px solid #3b82f6 !important; }
        .glow-trivial { border-bottom: 3px solid #16a34a !important; }
        
        .section-panel-title {
            font-size: 16px;
            font-weight: 600;
            color: #f1f5f9;
            margin-top: 5px;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Glass Input Control Center Panel Panel Setup */
        .control-glass-panel {
            background: rgba(13, 22, 37, 0.8);
            border: 1px solid #162235;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }

        div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"] select {
            background-color: #090e17 !important;
            border: 1px solid #1b2a41 !important;
            color: #f1f5f9 !important;
            border-radius: 8px !important;
            font-size: 14px;
        }

        .custom-table-container {
            overflow-y: auto; 
            max-height: 340px; 
            border: 1px solid #162235; 
            border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.5);
            background: rgba(11, 18, 31, 0.85);
            backdrop-filter: blur(12px);
        }
        
        .bug-log-table {
            width: 100%; 
            border-collapse: collapse; 
            font-family: sans-serif; 
            font-size: 13.5px;
            text-align: left;
        }
        
        .bug-log-table th {
            background-color: #0b121f; 
            border-bottom: 2px solid #162235; 
            color: #94a3b8; 
            padding: 14px 16px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. FIXED GLIDING BUG LOGO LAYER (With Premium Zig-Zag Flight Path)
# ==============================================================================
st.components.v1.html(
    """
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 120px; pointer-events: none; overflow: visible; z-index: 999999;">
        <style>
            @keyframes zigZagGlide {
                0% { 
                    left: -10%; 
                    top: 10px; 
                    transform: rotate(75deg); 
                }
                25% { 
                    top: 70px; /* Dips downwards */
                    transform: rotate(105deg); /* Tilts downstream */
                }
                50% { 
                    top: 10px; /* Pulls back upwards */
                    transform: rotate(75deg); /* Re-corrects vector */
                }
                75% { 
                    top: 70px; /* Dips downwards again */
                    transform: rotate(105deg); 
                }
                100% { 
                    left: 110%; 
                    top: 10px; 
                    transform: rotate(75deg); 
                }
            }
            .flying-bug-asset {
                position: absolute;
                font-size: 32px;
                animation: zigZagGlide 14s ease-in-out infinite;
                filter: drop-shadow(0 0 8px #22d3ee) drop-shadow(0 0 20px #3b82f6);
            }
        </style>
        <div class="flying-bug-asset">🪲</div>
    </div>
    """,
    height=120,
)

# ==============================================================================
# 4. HIGH-END LIVE INTERACTIVE JAVASCRIPT ANIMATED ENGINE BACKGROUND INJECTION
# ==============================================================================
st.html(
    """
    <canvas id="liveMatrixCanvas" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; background-color: #040810;"></canvas>
    <script>
        const canvas = document.getElementById('liveMatrixCanvas');
        const ctx = canvas.getContext('2d');

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        const particleArray = [];
        const numberOfParticles = 75;
        let mouseX = null;
        let mouseY = null;

        window.addEventListener('mousemove', (event) => {
            mouseX = event.x;
            mouseY = event.y;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2 + 1;
                this.speedX = (Math.random() - 0.5) * 0.6;
                this.speedY = (Math.random() - 0.5) * 0.6;
            }
            update() {
                this.x += this.speedX;
                this.y += this.speedY;

                if (this.x > canvas.width || this.x < 0) this.speedX = -this.speedX;
                if (this.y > canvas.height || this.y < 0) this.speedY = -this.speedY;

                if (mouseX && mouseY) {
                    let dx = mouseX - this.x;
                    let dy = mouseY - this.y;
                    let distance = Math.sqrt(dx*dx + dy*dy);
                    if (distance < 160) {
                        this.x += dx * 0.01;
                        this.y += dy * 0.01;
                    }
                }
            }
            draw() {
                ctx.fillStyle = 'rgba(59, 130, 246, 0.4)';
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        for (let i = 0; i < numberOfParticles; i++) {
            particleArray.push(new Particle());
        }

        function connectParticles() {
            for (let a = 0; a < particleArray.length; a++) {
                for (let b = a; b < particleArray.length; b++) {
                    let dx = particleArray[a].x - particleArray[b].x;
                    let dy = particleArray[a].y - particleArray[b].y;
                    let distance = Math.sqrt(dx*dx + dy*dy);

                    if (distance < 110) {
                        let opacity = (1 - (distance/110)) * 0.12;
                        ctx.strokeStyle = `rgba(96, 165, 250, ${opacity})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particleArray[a].x, particleArray[a].y);
                        ctx.lineTo(particleArray[b].x, particleArray[b].y);
                        ctx.stroke();
                    }
                }
            }
        }

        function animateLoop() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < particleArray.length; i++) {
                particleArray[i].update();
                particleArray[i].draw();
            }
            connectParticles();
            requestAnimationFrame(animateLoop);
        }
        animateLoop();
    </script>
    """
)

# ==============================================================================
# 5. HELPER TRANSFORMATION & DICTIONARY ENGINES
# ==============================================================================
def get_solution_meta(sev):
    solutions = {
        "blocker": "Fatal crash in UI login module... Check memory allocations.",
        "critical": "DB connection fails under high load... Validate connection pooling.",
        "major": "Functional search results incomplete... Optimize index rules.",
        "minor": "Medium low functional boundary adjustment exception logged.",
        "trivial": "Text overlap on profile page dashboard asset... Fix CSS alignment.",
    }
    return solutions.get(sev.lower(), "Review logging metrics for trace debugging.")

ROW_BACKGROUNDS = {
    "BLOCKER": "rgba(36, 16, 19, 0.7)", "CRITICAL": "rgba(38, 21, 12, 0.7)", "MAJOR": "rgba(36, 30, 10, 0.7)",
    "MINOR": "rgba(10, 19, 36, 0.7)", "TRIVIAL": "rgba(7, 24, 16, 0.7)"
}
ROW_TEXT_COLORS = {
    "BLOCKER": "#fecaca", "CRITICAL": "#fed7aa", "MAJOR": "#fde68a",
    "MINOR": "#bfdbfe", "TRIVIAL": "#bbf7d0"
}
BADGE_STYLES = {
    "BLOCKER": "background: #e11d48; color: #ffffff;",
    "CRITICAL": "background: #ea580c; color: #ffffff;",
    "MAJOR": "background: #eab308; color: #1e293b;",
    "MINOR": "background: #3b82f6; color: #ffffff;",
    "TRIVIAL": "background: #16a34a; color: #ffffff;"
}

# ==============================================================================
# 6. SIDEBAR NAVIGATION CONTROLS
# ==============================================================================
st.sidebar.markdown(
    "<h2 style='color:#ffffff; font-size:18px; font-weight:600; margin-bottom:2px; letter-spacing:0.5px;'>🪲 AI SEVERITY CORE</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='color:#475569; font-size:12px; margin-top:0; margin-bottom:25px;'>Workspace Navigation Center</p>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:2px;'>Navigate view:</p>", unsafe_allow_html=True)
menu = st.sidebar.radio("", ["🏠 Home Workspace", "🔄 Advanced Logs"], label_visibility="collapsed")

st.sidebar.markdown("<br><p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:5px;'>Upload Incident CSV File</p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None and "processed_file" not in st.session_state:
    df_upload = pd.read_csv(uploaded_file)
    if "Description" in df_upload.columns:
        if model and vectorizer:
            df_upload["clean"] = df_upload["Description"].apply(preprocess_text)
            X_batch = vectorizer.transform(df_upload["clean"])
            df_upload["Predicted_Severity"] = model.predict(X_batch)
        else:
            df_upload["Predicted_Severity"] = np.random.choice(["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "TRIVIAL"], size=len(df_upload))
            
        df_upload["Predicted_Severity"] = df_upload["Predicted_Severity"].str.upper().str.strip()
        df_upload["Solution"] = df_upload["Predicted_Severity"].apply(get_solution_meta)
        
        st.session_state.bug_data = pd.concat(
            [st.session_state.bug_data, df_upload[["Description", "Predicted_Severity", "Solution"]]], 
            ignore_index=True
        )
        st.session_state.processed_file = True

st.sidebar.markdown("<br>", unsafe_allow_html=True)
export_placeholder = st.sidebar.empty()

if not st.session_state.bug_data.empty:
    csv_bytes = st.session_state.bug_data.to_csv(index=False).encode("utf-8")
    export_placeholder.download_button(
        label="📥 Export Priority Sorted Sheet",
        data=csv_bytes,
        file_name="priority_sorted_bugs.csv",
        mime="text/csv",
        use_container_width=True
    )

st.sidebar.markdown("<br><br><br><br><br><hr style='border-color:#141f32;'>", unsafe_allow_html=True)
st.sidebar.caption("Production Build v2.0.0")

# ==============================================================================
# 7. DYNAMIC COMPUTATION MATRIX (KPI ENGINE)
# ==============================================================================
log_dataframe = st.session_state.bug_data.copy()
if not log_dataframe.empty:
    log_dataframe["Predicted_Severity"] = log_dataframe["Predicted_Severity"].str.upper().str.strip()

counts = log_dataframe["Predicted_Severity"].value_counts() if not log_dataframe.empty else {}

b_val = counts.get("BLOCKER", 0)
c_val = counts.get("CRITICAL", 0)
m_val = counts.get("MAJOR", 0)
mi_val = counts.get("MINOR", 0)
t_val = counts.get("TRIVIAL", 0)

# ==============================================================================
# 8. SPARKLINES GENERATION
# ==============================================================================
def make_sparkline_svg(color, seed_val):
    np.random.seed(seed_val)
    points = np.random.randint(10, 35, size=8)
    path_data = "M " + " L ".join([f"{i*14},{p}" for i, p in enumerate(points)])
    return f"""<svg width="100" height="40" style="opacity: 0.8; overflow: visible;"><path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

# ==============================================================================
# 9. MAIN VIEWPORT LAYOUT WRAPPING
# ==============================================================================
st.markdown("<div class='header-title-container'>🛸 AI BUG SEVERITY ANALYZER</div>", unsafe_allow_html=True)

html_metrics_string = f"""
<div class="glass-metric-row">
    <div class="glass-metric-card glow-blocker">
        <div class="metric-text-block">
            <span class="metric-label">BLOCKER</span>
            <span class="metric-digits">{b_val}</span>
        </div>
        {make_sparkline_svg('#e11d48', 42)}
    </div>
    <div class="glass-metric-card glow-critical">
        <div class="metric-text-block">
            <span class="metric-label">CRITICAL</span>
            <span class="metric-digits">{c_val}</span>
        </div>
        {make_sparkline_svg('#ea580c', 15)}
    </div>
    <div class="glass-metric-card glow-major">
        <div class="metric-text-block">
            <span class="metric-label">MAJOR</span>
            <span class="metric-digits">{m_val}</span>
        </div>
        {make_sparkline_svg('#eab308', 88)}
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
        {make_sparkline_svg('#16a34a', 61)}
    </div>
</div>
"""
st.markdown(html_metrics_string, unsafe_allow_html=True)

# ROW 2 LAYOUT: SPLIT CONTROLS & ANALYTICAL BREAKDOWN
mid_left_panel, mid_right_panel = st.columns([5, 7], gap="large")

with mid_left_panel:
    st.markdown("<div class='section-panel-title'>🕹️ Diagnostic Control Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='control-glass-panel'>", unsafe_allow_html=True)
    
    manual_input_entry = st.text_area(
        "Describe the bug...",
        height=95,
        placeholder="Enter raw stack traces, log dumps, or issue descriptions here...",
        label_visibility="collapsed",
        key="single_bug_textarea"
    )
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        classify_clicked = st.button("Classify & Solve Issue", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear Memory Banks", use_container_width=True):
            st.session_state.bug_data = pd.DataFrame(columns=["Description", "Predicted_Severity", "Solution"])
            if "processed_file" in st.session_state:
                del st.session_state.processed_file
            st.rerun()
            
    if classify_clicked and manual_input_entry.strip():
        if model and vectorizer:
            clean_text = preprocess_text(manual_input_entry)
            X_single = vectorizer.transform([clean_text])
            predicted_label = model.predict(X_single)[0].upper().strip()
        else:
            predicted_label = np.random.choice(["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "TRIVIAL"])
            
        calculated_solution = get_solution_meta(predicted_label)
        new_row = pd.DataFrame([{
            "Description": manual_input_entry,
            "Predicted_Severity": predicted_label,
            "Solution": calculated_solution
        }])
        st.session_state.bug_data = pd.concat([st.session_state.bug_data, new_row], ignore_index=True)
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)

with mid_right_panel:
    st.markdown("<div class='section-panel-title'>📊 Live Analytical Breakdown</div>", unsafe_allow_html=True)
    
    categories = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
    chart_values = [b_val, c_val, m_val, mi_val, t_val]
    premium_bar_colors = ['#e11d48', '#ea580c', '#eab308', '#3b82f6', '#16a34a']
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=chart_values,
        marker=dict(
            color=premium_bar_colors,
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        width=0.4,
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=165,
        xaxis=dict(tickfont=dict(color='#94a3b8', size=12), showgrid=False, fixedrange=True),
        yaxis=dict(tickfont=dict(color='#475569', size=10), gridcolor='#141f32', showgrid=True, fixedrange=True),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==============================================================================
# ROW 3 LAYOUT: CUSTOM HTML RENDERED DATA MATRIX LOG WITH HARD FILTER SEARCH
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
title_col, search_col = st.columns([7, 5])

with title_col:
    st.markdown("<div class='section-panel-title' style='margin-bottom: 0px; margin-top: 8px;'>📋 Priority Sorted Incident Bug Logs</div>", unsafe_allow_html=True)

with search_col:
    search_severity = st.selectbox(
        "Filter Data View by Severity Level:",
        options=["SHOW ALL LABELS", "BLOCKER", "CRITICAL", "MAJOR", "MINOR", "TRIVIAL"],
        index=0,
        label_visibility="collapsed"
    )

severity_order = {"BLOCKER": 1, "CRITICAL": 2, "MAJOR": 3, "MINOR": 4, "TRIVIAL": 5}

if not log_dataframe.empty:
    log_dataframe["rank_index"] = log_dataframe["Predicted_Severity"].map(severity_order).fillna(6)
    sorted_df = log_dataframe.sort_values(by="rank_index").copy()
    
    if search_severity != "SHOW ALL LABELS":
        sorted_df = sorted_df[sorted_df["Predicted_Severity"] == search_severity]

    if not sorted_df.empty:
        html_table = """
        <div class="custom-table-container">
            <table class="bug-log-table">
                <thead>
                    <tr>
                        <th style="width: 40%;">Bug Description</th>
                        <th style="width: 20%; text-align: center;">Severity Assessment</th>
                        <th style="width: 40%;">Recommended System Action Plan</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in sorted_df.iterrows():
            desc = str(row['Description']).strip()
            sev = str(row['Predicted_Severity']).strip().upper()
            sol = str(row['Solution']).strip()
            
            bg = ROW_BACKGROUNDS.get(sev, "rgba(10, 19, 36, 0.7)")
            tc = ROW_TEXT_COLORS.get(sev, "#cbd5e1")
            badge = BADGE_STYLES.get(sev, "background: #475569; color: #ffffff;")
            
            html_table += f"""
            <tr style="background-color: {bg}; color: {tc}; border-bottom: 1px solid rgba(20, 31, 50, 0.5);">
                <td style="padding: 14px 16px; line-height: 1.5; vertical-align: middle;">{desc}</td>
                <td style="padding: 14px 16px; text-align: center; vertical-align: middle;">
                    <span style="{badge} padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 11px; letter-spacing: 0.5px; display: inline-block; min-width: 90px;">
                        {sev}
                    </span>
                </td>
                <td style="padding: 14px 16px; line-height: 1.5; vertical-align: middle; color: #e2e8f0;">{sol}</td>
            </tr>
            """
            
        html_table += """
                </tbody>
            </table>
        </div>
        """
        st.html(html_table)
    else:
        st.html(
            f"""
            <div style="border: 1px dashed #e11d48; border-radius: 10px; padding: 45px; text-align: center; color: #94a3b8; font-size: 14px; background: rgba(36,16,19,0.5); backdrop-filter: blur(12px);">
                🔍 No matched bug rows found for filter metric: <b>{search_severity}</b>. Try selecting another label level.
            </div>
            """
        )
else:
    st.html(
        """
        <div style="border: 1px dashed #162235; border-radius: 10px; padding: 45px; text-align: center; color: #475569; font-size: 14px; background: rgba(11,18,31,0.6); backdrop-filter: blur(12px);">
            🚀 No active incidents loaded in current session context. Upload a CSV file or insert a description above to compute priority tracking vectors.
        </div>
        """
    )