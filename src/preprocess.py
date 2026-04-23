import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\n', ' ', text)       # remove line breaks
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove special characters
    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)