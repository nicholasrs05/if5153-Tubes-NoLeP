# recommendation_rag_service.py

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import joblib
import torch
import os
import gdown
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

RAG_INDEX_PATH = "../models/recommendation_rag_index.joblib"


def download_lora_adapter(adapter_dir):
    if not os.path.exists(adapter_dir):
        print(f"LoRA adapter not found. Downloading from Google Drive...")
        os.makedirs(adapter_dir, exist_ok=True)
        folder_id = "1cRi6jODHjCvI-lpOr2ZVIL7IDAFoHMbK"
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=adapter_dir, quiet=False, use_cookies=False)
        print(f"✓ LoRA adapter downloaded to: {adapter_dir}")
    else:
        print(f"LoRA adapter found locally: {adapter_dir}")


@dataclass
class RetrievedTreatmentDoc:
    disease: str
    treatment: str
    similarity: float


class RecommendationRAGService:
    def __init__(
        self,
        rag_index_path: str = RAG_INDEX_PATH,
        base_model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        adapter_path: str = None,
        device: str = None,
    ):
        # ----- Load RAG index -----
        print("Loading RAG index...")
        index = joblib.load(rag_index_path)
        self.diseases = index["diseases"]
        self.treatments = index["treatments"]
        self.combined_texts = index["combined_texts"]
        self.treatment_embeddings = np.array(index["treatment_embeddings"], dtype=np.float32)

        # ----- Load embedder -----
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(index["embedding_model_name"])

        # ----- Download LoRA adapter if needed -----
        if adapter_path:
            download_lora_adapter(adapter_path)

        # ----- Load fine-tuned Llama model -----
        print("Loading base model and tokenizer...")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True
        )

        # Load LoRA adapter if provided
        if adapter_path:
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("No adapter path provided, using base model only")
            self.model = base_model

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on: {self.device}")

    # ---------------- RETRIEVAL ----------------

    def _retrieve_treatments(
        self, 
        query_text: str, 
        top_k: int = 3
    ) -> List[RetrievedTreatmentDoc]:
        q_emb = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        # Compute cosine similarities
        sims = np.dot(self.treatment_embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [
            RetrievedTreatmentDoc(
                disease=self.diseases[i],
                treatment=self.treatments[i],
                similarity=float(sims[i]),
            )
            for i in top_idx
        ]

    # ---------------- GENERATION ----------------

    def _generate_recommendation(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the recommendation part after "### Recommendation:"
        if "### Recommendation:" in response:
            answer = response.split("### Recommendation:")[-1].strip()
        else:
            # Fallback: try to remove the prompt from response
            answer = response.replace(prompt, "").strip()

        return answer

    # ---------------- PUBLIC API ----------------

    def generate_recommendation(
        self,
        symptom: str,
        cause_or_disease: str,
        top_k_docs: int = 3,
        max_new_tokens: int = 100,
        use_rag_in_prompt: bool = False,
        predicted_category: str = None,
    ) -> Dict[str, Any]:
        # Build retrieval query combining symptom and cause
        retrieval_query = f"Symptoms: {symptom}\nCause/Disease: {cause_or_disease}"

        retrieved_docs = []

        # If use RAG and label is not emotional pain, include guidelines in prompt
        if use_rag_in_prompt and predicted_category != "Emotional pain":
            # Retrieve relevant treatment guidelines
            retrieved_docs = self._retrieve_treatments(retrieval_query, top_k=top_k_docs)
            
            guidelines_context = "\n\n".join(
                f"{i+1}. Disease: {doc.disease}\n   Treatment: {doc.treatment[:500]}..."
                for i, doc in enumerate(retrieved_docs)
            )
            
            alpaca_prompt = """Based on the patient's symptoms, cause/disease, and the medical treatment guidelines below, give a short recommendation to the patient (2 sentences max).

### Symptoms:
{symptom}

### Cause or Disease:
{cause_or_disease}

### Medical Treatment Guidelines:
{guidelines}

### Recommendation:
"""
            
            prompt = alpaca_prompt.format(
                symptom=symptom,
                cause_or_disease=cause_or_disease,
                guidelines=guidelines_context
            )
        else:
            enhanced_cause = cause_or_disease
            
            alpaca_prompt = """Based on the patient's symptoms and cause or disease mentioned, give a short recommendation to the patient (2 sentences max).

### Symptoms:
{symptom}

### Cause or Disease:
{cause_or_disease}

### Recommendation:
"""
            
            prompt = alpaca_prompt.format(
                symptom=symptom,
                cause_or_disease=enhanced_cause
            )

        # Generate personalized recommendation
        recommendation = self._generate_recommendation(
            prompt,
            max_new_tokens=max_new_tokens
        )

        return {
            "recommendation": recommendation,
            "retrieved_guidelines": [
                {
                    "disease": doc.disease,
                    "treatment": doc.treatment,
                    "similarity": doc.similarity
                }
                for doc in retrieved_docs
            ]
        }


# -------------- QUICK TEST --------------

if __name__ == "__main__":
    # Google Drive folder ID for LoRA adapter
    GDRIVE_FOLDER_ID = "1sYuN_9rF9YRjI0Cn1e-vRJ01V3aB-xyb"
    
    service = RecommendationRAGService(
        rag_index_path="../models/recommendation_rag_index.joblib",
        base_model_name="unsloth/Llama-3.2-1B-Instruct",
        adapter_path="../models/llama-recommendation-fine-tuned",
        adapter_gdrive_folder_id=GDRIVE_FOLDER_ID,
        device="cpu",
    )

    symptom = "My lower back has been aching constantly and gets worse when I sit for long periods."
    cause = "The symptoms indicate lower back pain, likely caused by poor posture or muscle strain."

    result = service.generate_recommendation(
        symptom=symptom,
        cause_or_disease=cause,
        top_k_docs=2,
        use_rag_in_prompt=False,
    )

    print("=" * 60)
    print("SYMPTOM:", symptom)
    print("\nCAUSE:", cause)
    print("\n=== Retrieved Treatment Guidelines ===")
    for i, doc in enumerate(result["retrieved_guidelines"], 1):
        print(f"\n{i}. Disease: {doc['disease']} (similarity: {doc['similarity']:.3f})")
        print(f"   Treatment: {doc['treatment'][:200]}...")

    print("\n=== Generated Recommendation ===")
    print(result["recommendation"])
    print("=" * 60)
