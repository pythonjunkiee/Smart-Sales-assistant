# chatbot.py
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('punkt', quiet=True)

FAQ_PATH = 'data/faq.csv'
VEC_PATH = 'models/faq_vec.joblib'
FAQ_DF = 'models/faq_df.joblib'

def train_chatbot():
    df = pd.read_csv(FAQ_PATH)
    questions = df['question'].values
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    vec.fit(questions)
    os.makedirs('models', exist_ok=True)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(df, FAQ_DF)
    print("Chatbot artifacts saved.")

def get_answer(user_query):
    vec = joblib.load(VEC_PATH)
    df = joblib.load(FAQ_DF)
    q_vec = vec.transform(df['question'].values)
    uq = vec.transform([user_query])
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(uq, q_vec)[0]
    best_idx = int(np.argmax(sims))
    score = sims[best_idx]
    if score < 0.2:
        return "Sorry, I couldn't find an exact answer. I can pass this to support."
    return df['answer'].iloc[best_idx]

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_chatbot()
