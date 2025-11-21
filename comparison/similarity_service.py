# similarity_service.py

from dataclasses import dataclass
from typing import List, Dict, Any
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


MODEL_PATH = "similarity_model.joblib"


@dataclass
class SimilarCase:
    index: int
    patient_comment: str
    patient_category: str
    similarity: float


class SimilarityService:
    def __init__(self, model_path: str = MODEL_PATH):
        # load model yang sudah di-train
        model = joblib.load(model_path)
        self.vectorizer = model["vectorizer"]
        self.X = model["X"]
        self.texts = model["texts"]
        self.labels = model["labels"]

    def retrieve_similar_cases(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[SimilarCase]:
        """
        Input:
            query_text    : keluhan baru (string)
            top_k         : jumlah kasus mirip yang dikembalikan
            min_similarity: batas minimal skor cosine (0-1)

        Output:
            List[SimilarCase]
        """
        # 1. Vectorize input
        q_vec = self.vectorizer.transform([query_text])

        # 2. Hitung cosine similarity dengan semua kasus di data
        sims = cosine_similarity(q_vec, self.X)[0]  # array shape (n_samples,)

        # 3. Ambil index top_k
        #    argsort ascending → [paling kecil ... paling besar]
        #    [::-1] untuk membalik → paling besar di depan
        top_indices = np.argsort(sims)[::-1]

        results: List[SimilarCase] = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < min_similarity:
                continue
            results.append(
                SimilarCase(
                    index=int(idx),
                    patient_comment=self.texts[idx],
                    patient_category=self.labels[idx],
                    similarity=score,
                )
            )
            if len(results) >= top_k:
                break

        return results


# Contoh pemakaian modul ini secara standalone
if __name__ == "__main__":
    service = SimilarityService()

    query = "My lower back hurts when I stand up"
    similar_cases = service.retrieve_similar_cases(query, top_k=3)

    print("Query:", query)
    print("Top 3 similar cases:\n")
    for case in similar_cases:
        print(f"[{case.similarity:.3f}] ({case.patient_category}) {case.patient_comment}")
