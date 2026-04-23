from src.data_loader import load_data
from src.preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
train, test = load_data()

# Preprocess text
X_train = train['Description'].apply(preprocess_text)
y_train = train['Label']

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Save model
joblib.dump(model, "models/severity_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model trained successfully with new dataset!")