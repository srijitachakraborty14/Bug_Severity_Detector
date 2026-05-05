import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Keep numbers (0-9) to catch error codes like 404 or 500
    text = re.sub(r'[^a-z0-9]', ' ', text)
    words = text.split()
    # Lemmatize words to find the root meaning
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)
