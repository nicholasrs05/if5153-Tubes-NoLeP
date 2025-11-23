# prepare_cause_rag_index.py

import pandas as pd
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer

CAUSE_KB_PATH = "Cause_Knowledge_EN.csv"          # <- kamu buat sendiri file ini
RAG_INDEX_PATH = "cause_rag_index.joblib"      # <- output index


def build_cause_rag_index(
    kb_path: str = CAUSE_KB_PATH,
    index_path: str = RAG_INDEX_PATH,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    # 1. Load knowledge base
    df = pd.read_csv(kb_path)
    df = df.dropna(subset=["Category", "CauseText"])

    categories = df["Category"].astype(str).tolist()
    cause_texts = df["CauseText"].astype(str).tolist()

    # 2. Load embedding model
    embedder = SentenceTransformer(embedding_model_name)

    # 3. Buat embeddings untuk setiap teks penyebab
    print("Encoding cause texts...")
    cause_embeddings = embedder.encode(
        cause_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # penting untuk cosine
    )

    index = {
        "embedding_model_name": embedding_model_name,
        "categories": categories,
        "cause_texts": cause_texts,
        "cause_embeddings": cause_embeddings,
    }

    joblib.dump(index, index_path)
    print(f"RAG index saved to: {os.path.abspath(index_path)}")


if __name__ == "__main__":
    build_cause_rag_index()
