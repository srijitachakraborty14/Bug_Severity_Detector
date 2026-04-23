from src.data_loader import load_data
from src.preprocess import preprocess_text
from sklearn.metrics import accuracy_score, classification_report
import joblib

train, test = load_data()

model = joblib.load("models/severity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

X_test = test['Description'].apply(preprocess_text)
y_test = test['Label']

X_test_tfidf = vectorizer.transform(X_test)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))