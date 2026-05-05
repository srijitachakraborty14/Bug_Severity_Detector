import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from src.data_loader import load_data
from src.preprocess import preprocess_text

def train_model():
    print("--- Starting Model Training Workflow ---")

    # 1. Load Data
    print("Step 1: Loading data...")
    train, test = load_data()

    # 2. Preprocess Text
    print("Step 2: Preprocessing text...")
    X_train = train['Description'].apply(preprocess_text)
    y_train = train['Label']

    # 3. Vectorization
    print("Step 3: Vectorizing text features...")
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # 4. Initialize Model
    model = LinearSVC(
        C=1.0, 
        random_state=42, 
        max_iter=2000,
        class_weight='balanced'
    )

    # 5. CROSS-VALIDATION (The Quality Check)
    # We do this before the final fit to validate the algorithm's stability
    print(f"Step 4: Running 5-Fold Cross-Validation on {len(y_train)} samples...")
    # cv=5 means the data is split into 5 parts, and we train 5 times
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, n_jobs=-1)
    
    print("\n--- Cross-Validation Results ---")
    print(f"Individual Fold Accuracies: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    print("--------------------------------\n")

    # 6. Final Fit and Save
    print("Step 5: Training final model on full training set...")
    model.fit(X_train_tfidf, y_train)

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(model, "models/severity_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("[SUCCESS] Model and Vectorizer saved.")

if __name__ == "__main__":
    train_model()