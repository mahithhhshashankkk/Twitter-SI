import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import load_and_clean_data
from src.vectorizer import extract_features
from src.model import train_model
from src.evaluate import evaluate_model
from src.predict import predict_texts
from config import TEST_SIZE, RANDOM_SEED

# Load and preprocess
df = load_and_clean_data('data/your_dataset.csv')
X, vectorizer = extract_features(df['text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Train model
model = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)

# Predict sample inputs
samples = [
    "I don’t want to live anymore",
    "I feel much better today",
    "I’m thinking about ending it all",
    "Life is beautiful and worth living"
]
preds = predict_texts(model, vectorizer, samples)

for s, p in zip(samples, preds):
    print(f"\nText: {s}\nPrediction: {'Suicidal' if p == 1 else 'Not Suicidal'}")

