from sklearn.feature_extraction.text import TfidfVectorizer
from config import TFIDF_PARAMS

def extract_features(texts):
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

