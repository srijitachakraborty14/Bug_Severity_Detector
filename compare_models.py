import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from nltk.corpus import stopwords

# stopwords
stop_words = set(stopwords.words("english"))


# ======================
# TEXT PREPROCESS FUNCTION
# ======================

def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)


# ======================
# LOAD DATASET
# ======================

df = pd.read_csv("data/raw/bug_dataset_50k.csv")

df = df[
    [
        "title",
        "description",
        "bug_category",
        "bug_domain",
        "tech_stack",
        "environment",
        "developer_role",
        "severity",
    ]
].dropna()


# combine columns

df["text"] = (
    df["title"] + " "
    + df["description"] + " "
    + df["bug_category"] + " "
    + df["bug_domain"] + " "
    + df["tech_stack"] + " "
    + df["environment"] + " "
    + df["developer_role"]
)


X = df["text"].apply(preprocess_text)
y = df["severity"].str.lower()


# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================
# TF-IDF VECTORIZE
# ======================

vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1, 2),
    stop_words="english",
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ======================
# MODELS TO TEST
# ======================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "SVM":
        LinearSVC(class_weight="balanced"),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier(n_estimators=200)

}


# ======================
# TRAIN + EVALUATE
# ======================

results = []

for name, model in models.items():

    print("\n==============================")
    print(f"Training: {name}")
    print("==============================")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("F1-score:", f1)

    print("\nDetailed Report:\n")
    print(classification_report(y_test, y_pred))

    results.append((name, accuracy, f1))


# ======================
# FINAL SUMMARY TABLE
# ======================

print("\n\nFINAL MODEL COMPARISON")

for name, acc, f1 in results:

    print(f"{name} --> Accuracy: {acc:.4f}, F1-score: {f1:.4f}")