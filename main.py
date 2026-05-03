#from src.inference import predict

# Take input from user
#text = input("Enter bug description: ")

# Predict severity and solution
#severity, solution = predict(text)

# Show results
#print("\nPredicted Severity:", severity)
#print("Suggested Solution:", solution)
import joblib
from src.preprocess import preprocess_text

# Load saved model and vectorizer
model = joblib.load("models/severity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Take user input
text = input("Enter bug description: ")

# Preprocess input
clean_text = preprocess_text(text)

# Convert to TF-IDF
vector = vectorizer.transform([clean_text])

# Predict severity
prediction = model.predict(vector)

print("Predicted Severity:", prediction[0])