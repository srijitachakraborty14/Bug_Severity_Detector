from src.data_loader import load_data
from src.preprocess import preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import joblib


def train_model(): #train

    train, test = load_data()

    X_train = train["Description"].apply(preprocess_text)
    y_train = train["Label"]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LinearSVC(class_weight="balanced")

    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, "models/severity_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("✅ Model trained successfully!")


if __name__ == "__main__":
    train_model()