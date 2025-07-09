from sklearn.linear_model import LogisticRegression
from config import RANDOM_SEED, MAX_ITER

def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=MAX_ITER, 
        class_weight='balanced', 
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model

