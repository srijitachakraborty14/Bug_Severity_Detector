import joblib
from src.preprocess import preprocess_text
from src.solution import get_solution

model = joblib.load("models/severity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict(text):

    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])

    severity = model.predict(vector)[0]

    solution = get_solution(text, severity)

    return severity, solution