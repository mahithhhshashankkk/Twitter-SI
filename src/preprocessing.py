import pandas as pd
import re

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.rename(columns={'Tweet': 'text', 'Suicide': 'label'}, inplace=True)
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].str.lower().apply(lambda x: re.sub(r'\W+', ' ', x))
    return df


