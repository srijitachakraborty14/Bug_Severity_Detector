import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) #stopword removal


def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"\n", " ", text)

    text = re.sub(r"[^a-zA-Z]", " ", text) #cleaned process

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)