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
# CUSTOM CSS
# ======================
st.markdown("""
<style>
.block-container {
    padding: 2rem;
}
h1 {
    text-align: center;
    color: #1e3a8a;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.title("🐞 AI Bug Severity Analyzer")

# ======================
# SINGLE INPUT SECTION
# ======================
st.subheader("🔍 Enter Bug Description")

text = st.text_area("Describe the bug in detail...")

if st.button("Predict Severity"):
    if len(text.strip()) == 0:
        st.warning("Please enter a bug description.")
    else:
        # Handle short input
        if len(text.split()) < 5:
            text = "System issue: " + text + " causing failure in application functionality"

        clean = preprocess_text(text)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]

        st.success(f"Predicted Severity: {pred}")

# ======================
# FILE UPLOAD SECTION
# ======================
st.subheader("📂 Upload Bug File (CSV)")

uploaded_file = st.file_uploader("Upload CSV with 'Description' column", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Check column
    if "Description" not in df.columns:
        st.error("CSV must contain 'Description' column")
    else:
        st.write("📄 Uploaded Data")
        st.dataframe(df.head())

        # ======================
        # PREDICTION
        # ======================
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
        # DISPLAY TABLE
        # ======================
        st.subheader("📊 Predicted Bugs (Sorted)")
        st.dataframe(df[["Description", "Predicted_Severity", "Solution"]])

        # ======================
        # DOWNLOAD BUTTON
        # ======================
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Results",
            csv,
            "predicted_bugs.csv",
            "text/csv"
        )

        # ======================
        # CHART
        # ======================
        st.subheader("📈 Severity Distribution")
        st.bar_chart(df["Predicted_Severity"].value_counts())