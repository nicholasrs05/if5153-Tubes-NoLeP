# cause_generation_service.py (Qwen, English prompt, stop on period)

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import joblib
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)


RAG_INDEX_PATH = "cause_rag_index.joblib"


@dataclass
class RetrievedCauseDoc:
    category: str
    cause_text: str
    similarity: float


class StopOnPeriod(StoppingCriteria):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_token_len: int,
        min_generated_tokens: int = 5,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
        self.min_generated_tokens = min_generated_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        # input_ids shape: (batch_size, seq_len)
        full_ids = input_ids[0]

        # only look at generated part (exclude prompt tokens)
        gen_ids = full_ids[self.prompt_token_len :]
        if gen_ids.shape[0] < self.min_generated_tokens:
            return False

        # decode only generated tokens
        gen_text = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if not gen_text:
            return False

        # check last non-space character
        last_char = gen_text.rstrip()[-1]
        return last_char == "."


class CauseRAGService:
    def __init__(
        self,
        rag_index_path: str = RAG_INDEX_PATH,
        generator_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cpu",
    ):
        # ----- Load RAG index -----
        index = joblib.load(rag_index_path)
        self.categories = index["categories"]
        self.cause_texts = index["cause_texts"]
        self.cause_embeddings = np.array(index["cause_embeddings"], dtype=np.float32)

        # ----- Load embedder -----
        self.embedder = SentenceTransformer(index["embedding_model_name"])

        # ----- Load Qwen causal LM -----
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

    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        min_generated_tokens: int = 5,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        prompt_token_len = inputs["input_ids"].shape[1]

        stopping = StoppingCriteriaList(
            [StopOnPeriod(self.tokenizer, prompt_token_len, min_generated_tokens)]
        )

        with torch.no_grad():
            output_ids = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # strip prompt
        if full_text.startswith(prompt):
            gen_text = full_text[len(prompt) :].strip()
        else:
            gen_text = full_text.strip()

        return gen_text

    # ---------------- PUBLIC API ----------------

    def generate_cause_explanation(
        self,
        complaint_text: str,
        predicted_category: Optional[str] = None,
        top_k_docs: int = 3,
        max_new_tokens: int = 80,
    ) -> Dict[str, Any]:
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
- The answer MUST be exactly one coherent sentence in English.
- Explain how the patient's activity and condition might be related to the possible causes.
- Use simple, patient-friendly language.

Patient complaint:
{complaint_text}

Predicted category (from classification model):
{predicted_category}

Relevant medical knowledge:
{kb_context}

Now write ONE sentence explaining the possible causes:
"""

        cause_text = self._generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            min_generated_tokens=5,
        )

        return {
            "cause_explanation": cause_text,
            "retrieved_docs": [d.__dict__ for d in retrieved_docs],
        }


# -------------- QUICK TEST --------------

if __name__ == "__main__":
    service = CauseRAGService(
        rag_index_path=RAG_INDEX_PATH,
        generator_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",
    )

    complaint = "My lower back hurts a lot after I lift heavy things at work."
    predicted_category = "Back pain"

    out = service.generate_cause_explanation(
        complaint_text=complaint,
        predicted_category=predicted_category,
        top_k_docs=2,
        max_new_tokens=40,
    )

    print("Complaint:", complaint)
    print("\n=== Retrieved docs ===")
    for d in out["retrieved_docs"]:
        print(f"- ({d['category']}, sim={d['similarity']:.3f}) {d['cause_text']}")

    print("\n=== Generated cause explanation ===")
    print(out["cause_explanation"])
