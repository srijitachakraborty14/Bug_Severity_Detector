import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data_loader import load_data
from src.preprocess import preprocess_text


def evaluate_model():
    # Load raw validation data
    _, test = load_data()

    # Load saved configurations
    model = joblib.load("models/severity_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    # Transform test features using the trained space map
    X_test = test["Description"].apply(preprocess_text)
    y_test = test["Label"]

    X_test_tfidf = vectorizer.transform(X_test)

    # Inference
    y_pred = model.predict(X_test_tfidf)

    # Outputs
    print("\n================ EVALUATION RESULTS ================")
    print("Overall Test Accuracy:", accuracy_score(y_test, y_pred))

    print("\nDetailed Per-Class Performance Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nRaw Confusion Matrix Layout:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()