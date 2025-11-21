# similarity_train.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

DATA_PATH = "data/HealthCare Data.csv"          # sesuaikan dengan lokasi file kamu
MODEL_PATH = "models/similarity_model.joblib"     # file output model similarity


def train_similarity_model(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
):
    # 1. Load data
    df = pd.read_csv(data_path, encoding="latin-1")  # encoding bisa disesuaikan
    texts = df["Patient_comment"].astype(str)
    labels = df["Patient_Category"].astype(str)

    # 2. Vectorizer: TF-IDF ngram (1-2)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2,
        stop_words="english",  # bisa dikosongkan kalau banyak teks non-English
    )

    X = vectorizer.fit_transform(texts)  # shape: (n_samples, n_features)

    # 3. Simpan semua yang dibutuhkan untuk inference
    model = {
        "vectorizer": vectorizer,
        "X": X,
        "texts": texts.tolist(),
        "labels": labels.tolist(),
    }

    joblib.dump(model, model_path)
    print(f"Similarity model saved to: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    train_similarity_model()
