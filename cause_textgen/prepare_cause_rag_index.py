import pandas as pd
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer


CAUSE_KB_PATH = "../data/cause_from_pdf.csv"
RAG_INDEX_PATH = "../models/cause_rag_index.joblib"


def build_cause_rag_index(
    kb_path: str = CAUSE_KB_PATH,
    index_path: str = RAG_INDEX_PATH,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    # 1. Load knowledge base from CSV
    print(f"[INFO] Loading CSV knowledge base from: {kb_path}")
    df = pd.read_csv(kb_path)

    if not {"Category", "CauseText"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Category' and 'CauseText' columns.")

    df = df.dropna(subset=["Category", "CauseText"])

    categories = df["Category"].astype(str).tolist()
    cause_texts = df["CauseText"].astype(str).tolist()

    print(f"[INFO] Loaded {len(cause_texts)} rows from CSV.")

    # 2. Load embedding model
    print(f"[INFO] Loading embedding model: {embedding_model_name}")
    embedder = SentenceTransformer(embedding_model_name)

    # 3. Encode cause texts
    print("[INFO] Encoding cause texts...")
    cause_embeddings = embedder.encode(
        cause_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    index = {
        "embedding_model_name": embedding_model_name,
        "categories": categories,
        "cause_texts": cause_texts,
        "cause_embeddings": cause_embeddings.astype(np.float32),
    }

    joblib.dump(index, index_path)
    print(f"[INFO] RAG index saved to: {os.path.abspath(index_path)}")


if __name__ == "__main__":
    build_cause_rag_index()
