import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src.data_loader import load_data
from src.preprocess import preprocess_text


def train_model():
    # Load dataset
    train, test = load_data()

    # ======================
    # DEBUGGING INFO
    # ======================
    print("\n===== TRAINING LABEL DISTRIBUTION =====")
    print(train["Label"].value_counts())

    # ======================
    # PREPROCESSING
    # ======================
    # Applies your custom preprocess cleaning over the engineered text
    X_train = train["Description"].apply(preprocess_text)
    y_train = train["Label"]

    # ======================
    # TF-IDF VECTORIZATION
    # ======================
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),  # Captures key context combinations (ex: "crash on")
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    # ======================
    # ADJUSTED BALANCED MODEL
    # ======================
    # C=0.5 prevents the model from ignoring class weights to minimize global training loss
    model = LinearSVC(class_weight="balanced", C=0.5, random_state=42)

    print("\nTraining LinearSVC model with optimized class boundaries...")
    model.fit(X_train_tfidf, y_train)

    # ======================
    # SAVE ARTIFACTS
    # ======================
    joblib.dump(model, "models/severity_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("\n✅ Model trained successfully!")
    print("✅ Model saved to models/severity_model.pkl")
    print("✅ Vectorizer saved to models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    train_model()