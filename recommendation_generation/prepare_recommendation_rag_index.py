# prepare_recommendation_rag_index.py

import pandas as pd
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer

TREATMENT_KB_PATH = "standard-treatment-guidelines.csv"
RAG_INDEX_PATH = "../models/recommendation_rag_index.joblib"


def build_recommendation_rag_index(
    kb_path: str = TREATMENT_KB_PATH,
    index_path: str = RAG_INDEX_PATH,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Build RAG index from treatment guidelines CSV.
    Combines disease name and treatment text for embedding.
    """
    # 1. Load knowledge base
    print(f"Loading treatment guidelines from: {kb_path}")
    df = pd.read_csv(kb_path)
    df = df.dropna(subset=["disease", "treatment"])

    diseases = df["disease"].astype(str).tolist()
    treatments = df["treatment"].astype(str).tolist()

    # 2. Create combined text for better retrieval
    # Combine disease name with treatment for semantic matching
    combined_texts = [
        f"Disease: {disease}\nTreatment: {treatment}"
        for disease, treatment in zip(diseases, treatments)
    ]

    # 3. Load embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    embedder = SentenceTransformer(embedding_model_name)

    # 4. Create embeddings for each treatment guideline
    print("Encoding treatment guidelines...")
    treatment_embeddings = embedder.encode(
        combined_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity
    )

    # 5. Build and save index
    index = {
        "embedding_model_name": embedding_model_name,
        "diseases": diseases,
        "treatments": treatments,
        "combined_texts": combined_texts,
        "treatment_embeddings": treatment_embeddings,
    }

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    joblib.dump(index, index_path)
    print(f"✓ RAG index saved to: {os.path.abspath(index_path)}")
    print(f"✓ Indexed {len(diseases)} treatment guidelines")


if __name__ == "__main__":
    build_recommendation_rag_index()
