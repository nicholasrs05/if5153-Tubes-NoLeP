"""
RAG-enhanced inference script for medication recommendation.
This script demonstrates how to use the RecommendationRAGService.
"""

from recommendation_rag_service import RecommendationRAGService

# Initialize RAG service
print("Initializing RAG-based Recommendation Service...")
service = RecommendationRAGService(
    rag_index_path="../models/recommendation_rag_index.joblib",
    base_model_name="unsloth/Llama-3.2-1B-Instruct",
    adapter_path=None,  # TODO: Update with your LoRA adapter path when available
    device="cpu",  # Change to "cuda" if GPU available
)

# Inference function with RAG
def generate_recommendation(symptom, cause_or_disease, top_k_guidelines=3, use_rag_in_prompt=False):
    result = service.generate_recommendation(
        symptom=symptom,
        cause_or_disease=cause_or_disease,
        top_k_docs=top_k_guidelines,
        max_new_tokens=80,
        use_rag_in_prompt=use_rag_in_prompt,
    )
    return result


# Test
if __name__ == "__main__":
    symptom = "My lower back has been aching constantly and gets worse when I sit for long periods."
    cause_or_disease = "The symptoms indicate lower back pain, likely caused by poor posture or muscle strain."

    print("\n" + "=" * 60)
    print("Testing RAG-based Recommendation Generation")
    print("=" * 60)
    
    result = generate_recommendation(symptom, cause_or_disease, top_k_guidelines=2)
    
    print(f"\nSymptom: {symptom}")
    print(f"\nCause: {cause_or_disease}")
    
    print("\n--- Retrieved Treatment Guidelines ---")
    for i, doc in enumerate(result["retrieved_guidelines"], 1):
        print(f"\n{i}. Disease: {doc['disease']} (similarity: {doc['similarity']:.3f})")
        print(f"   Treatment snippet: {doc['treatment'][:150]}...")
    
    print("\n--- Generated Recommendation ---")
    print(result["recommendation"])
    print("\n" + "=" * 60)