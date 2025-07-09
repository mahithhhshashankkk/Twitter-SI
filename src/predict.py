import re

def preprocess_text(texts):
    return [re.sub(r'\W+', ' ', t.lower()) for t in texts]

def predict_texts(model, vectorizer, texts):
    processed = preprocess_text(texts)
    X = vectorizer.transform(processed)
    return model.predict(X)

