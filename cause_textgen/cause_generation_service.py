# cause_generation_service.py (Qwen, English prompt)

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import joblib
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


RAG_INDEX_PATH = "cause_rag_index.joblib"


@dataclass
class RetrievedCauseDoc:
    category: str
    cause_text: str
    similarity: float


class CauseRAGService:
    """
    RAG-based cause explanation generator using:
    - SentenceTransformers for retrieval
    - Qwen3-4B-Instruct-2507 for generation (causal LM)
    """

    def __init__(
        self,
        rag_index_path: str = RAG_INDEX_PATH,
        generator_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cpu",
    ):
        # Load RAG index
        index = joblib.load(rag_index_path)
        self.categories = index["categories"]
        self.cause_texts = index["cause_texts"]
        self.cause_embeddings = np.array(index["cause_embeddings"], dtype=np.float32)

        # Load embedder (same as used when building index)
        self.embedder = SentenceTransformer(index["embedding_model_name"])

        # Load Qwen causal LM
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            generator_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.gen_model.to(self.device)
        self.gen_model.eval()

    # ---------------- RETRIEVAL ----------------

    def _retrieve_docs(self, query_text: str, top_k: int = 3) -> List[RetrievedCauseDoc]:
        q_emb = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        sims = np.dot(self.cause_embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [
            RetrievedCauseDoc(
                category=self.categories[i],
                cause_text=self.cause_texts[i],
                similarity=float(sims[i]),
            )
            for i in top_idx
        ]

    # ---------------- GENERATION (QWEN) ----------------

    def _generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate text using Qwen (causal LM).
        """
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.gen_model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # strip the prompt from the decoded text if it’s echoed
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        return output_text.strip()

    # ---------------- PUBLIC API ----------------

    def generate_cause_explanation(
        self,
        complaint_text: str,
        predicted_category: Optional[str] = None,
        top_k_docs: int = 3,
        max_new_tokens: int = 200,
    ) -> Dict[str, Any]:
        """
        Generate an English explanation of possible medical causes
        for the patient's complaint using RAG + Qwen.
        """

        # Build retrieval query (English)
        if predicted_category:
            retrieval_query = (
                f"Complaint: {complaint_text}\n"
                f"Category: {predicted_category}"
            )
        else:
            retrieval_query = complaint_text

        retrieved_docs = self._retrieve_docs(retrieval_query, top_k=top_k_docs)

        kb_context = "\n".join(
            f"{i+1}. [Category: {d.category}] {d.cause_text}"
            for i, d in enumerate(retrieved_docs)
        )

        # -------- ENGLISH PROMPT FOR QWEN --------
        prompt = f"""
You are a medical assistant. Your task is to explain possible medical causes
for the patient's symptoms, using the complaint and the retrieved medical knowledge.

REQUIREMENTS:
- Write the answer in ENGLISH.
- The answer MUST be one coherent paragraph with only 1 sentences.
- DO NOT give a definite diagnosis or prescribe specific drugs.
- Use simple, patient-friendly language.
- Explain how the patient's activity and condition might be related to the possible causes.

Patient complaint:
{complaint_text}

Predicted category (from classification model):
{predicted_category}

Relevant medical knowledge:
{kb_context}

Now write a short paragraph explaining the possible causes:
"""

        cause_text = self._generate_text(prompt, max_new_tokens=max_new_tokens)

        return {
            "cause_explanation": cause_text,
            "retrieved_docs": [d.__dict__ for d in retrieved_docs],
        }


# -------------- QUICK TEST --------------

if __name__ == "__main__":
    service = CauseRAGService(
        rag_index_path=RAG_INDEX_PATH,
        generator_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",  # change to "cuda" if you have GPU
    )

    complaint = "My lower back hurts a lot after I lift heavy things at work."
    predicted_category = "Back pain"

    out = service.generate_cause_explanation(
        complaint_text=complaint,
        predicted_category=predicted_category,
        top_k_docs=2,
    )

    print("Complaint:", complaint)
    print("\n=== Retrieved docs ===")
    for d in out["retrieved_docs"]:
        print(f"- ({d['category']}, sim={d['similarity']:.3f}) {d['cause_text']}")

    print("\n=== Generated cause explanation ===")
    print(out["cause_explanation"])
