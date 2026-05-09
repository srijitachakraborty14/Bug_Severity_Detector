import streamlit as st
import pandas as pd
import joblib

from src.preprocess import preprocess_text

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="AI Bug Severity Analyzer", layout="wide")

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/severity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ======================
# SIDEBAR
# ======================
st.sidebar.markdown("## 🐞 Bug Analyzer")
menu = st.sidebar.radio("", ["🏠 Home", "📂 Upload", "📊 Reports"])

# ======================
# HEADER
# ======================
st.markdown(
    "<h1 style='color:white; text-align:left;'>AI Bug Severity Analyzer</h1>",
    unsafe_allow_html=True
)

# ======================
# MAIN LAYOUT
# ======================
left, right = st.columns([2, 1])

# ======================
# LEFT PANEL
# ======================
with left:
    st.subheader("Enter Bug Description")

    text = st.text_area("Describe the bug in detail...", height=150)

    if st.button("✨ Predict"):
        if len(text.strip()) == 0:
            st.warning("Please enter a bug description.")
        else:
            if len(text.split()) < 5:
                text = "System issue: " + text + " causing failure in application functionality"

            clean = preprocess_text(text)
            vec = vectorizer.transform([clean])
            pred = model.predict(vec)[0]

            st.success(f"Predicted Severity: {pred.upper()}")

    # Upload
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV with 'Description' column", type=["csv"])

# ======================
# RIGHT PANEL (STATIC SUMMARY)
# ======================
with right:
    st.subheader("Prediction Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BLOCKER", "—")
    c2.metric("CRITICAL", "—")
    c3.metric("MAJOR", "—")
    c4.metric("MINOR", "—")
    c5.metric("TRIVIAL", "—")

# ======================
# FILE PROCESSING
# ======================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Description" not in df.columns:
        st.error("CSV must contain 'Description' column")
    else:
        # Prediction
        df["clean"] = df["Description"].apply(preprocess_text)
        X = vectorizer.transform(df["clean"])
        df["Predicted_Severity"] = model.predict(X)

        # ======================
        # SOLUTION MAPPING
        # ======================
        def get_solution(sev):
            solutions = {
                "blocker": "🚨 Fix immediately. System unusable.",
                "critical": "⚠️ High priority fix required.",
                "major": "🔧 Fix in next release.",
                "minor": "🛠️ Low priority issue.",
                "trivial": "🎨 Cosmetic issue."
            }
            return solutions.get(sev, "No solution")

        df["Solution"] = df["Predicted_Severity"].apply(get_solution)

        # ======================
        # SORTING
        # ======================
        severity_order = {
            "blocker": 1,
            "critical": 2,
            "major": 3,
            "minor": 4,
            "trivial": 5
        }

        df["priority"] = df["Predicted_Severity"].map(severity_order)
        df = df.sort_values(by="priority")

        # ======================
        # SUMMARY COUNTS
        # ======================
        counts = df["Predicted_Severity"].value_counts()

        with right:
            st.subheader("Updated Summary")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("BLOCKER", counts.get("blocker", 0))
            c2.metric("CRITICAL", counts.get("critical", 0))
            c3.metric("MAJOR", counts.get("major", 0))
            c4.metric("MINOR", counts.get("minor", 0))
            c5.metric("TRIVIAL", counts.get("trivial", 0))

        # ======================
        # CHART
        # ======================
        st.subheader("📊 Severity Distribution")
        st.bar_chart(counts)

        # ======================
        # COLOR FUNCTION (FIXED)
        # ======================
        def color_severity(val):
            colors = {
                "blocker": "background-color:#fecaca",
                "critical": "background-color:#fed7aa",
                "major": "background-color:#fde68a",
                "minor": "background-color:#bfdbfe",
                "trivial": "background-color:#bbf7d0"
            }
            return colors.get(str(val).lower(), "")

        # ======================
        # TABLE (FIXED)
        # ======================
        st.subheader("📋 Predicted Bugs")

        styled_df = df[["Description", "Predicted_Severity", "Solution"]].style.map(
            color_severity,
            subset=["Predicted_Severity"]
        )

        st.dataframe(styled_df)

        # ======================
        # DOWNLOAD
        # ======================
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Export Results",
            csv,
            "results.csv",
            "text/csv"
        )