import joblib
from src.preprocess import preprocess_text

# Load trained model and vectorizer
model = joblib.load("models/severity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Take user input
text = input("Enter bug description: ")

# 🔧 Fix for short inputs
if len(text.split()) < 5:
    text = "System issue: " + text + " causing failure in application functionality"

# Preprocess text
clean = preprocess_text(text)

# Convert to TF-IDF
vec = vectorizer.transform([clean])

# Predict severity
prediction = model.predict(vec)

# Output result
print("Predicted Severity:", prediction[0])