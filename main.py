from classification.predictions import classify_single_text, load_model
from cause_textgen.cause_generation_service import CauseRAGService
from recommendation_generation.recommendation_rag_service import RecommendationRAGService



def main():
    print("Loading models... please wait.")

    # ---- Load classification model ----
    model_classification_path = "models/roberta_classification"
    model, tokenizer, id2label, device = load_model(model_classification_path)

    cause_generation_service = CauseRAGService(
        rag_index_path="models/cause_rag_index.joblib",
        generator_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",
    )

    # ---- Load recommendation generation service (RAG-based) ----
    recommendation_service = RecommendationRAGService(
        rag_index_path="models/recommendation_rag_index.joblib",
        base_model_name="unsloth/Llama-3.2-1B-Instruct",
        adapter_path="models/llama-recommendation-fine-tuned-150",
        device="cpu",
    )

    print("Models loaded! You can now start the pipeline.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Enter patient complaint: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the program.")
                break

            if not user_input:
                print("Empty input, please type a complaint sentence.\n")
                continue

            # -------------------------------------------------
            # 1) CLASSIFICATION STEP
            # -------------------------------------------------
            print("\n[STEP 1] Classification")
            predicted_label = classify_single_text(
                user_input, model, tokenizer, id2label, device
            )
            print(f"Predicted label: {predicted_label}")

            # -------------------------------------------------
            # 2) CAUSE GENERATION STEP
            # -------------------------------------------------
            print("\n[STEP 2] Cause Generation (RAG + Qwen)")
            cause_result = cause_generation_service.generate_cause_explanation(
                complaint_text=user_input,
                predicted_category=predicted_label,
                top_k_docs=2,
                max_new_tokens=40,
            )

            cause_explanation = cause_result["cause_explanation"]
            print("Generated cause explanation:")
            print(cause_explanation)

            # (Optional) Kalau mau lihat dokumen RAG yang terambil:
            # print("\nRetrieved docs:")
            # for d in cause_result["retrieved_docs"]:
            #     print(f"- ({d['category']}, sim={d['similarity']:.3f}) {d['cause_text']}")

            # -------------------------------------------------
            # 3) RECOMMENDATION GENERATION STEP (RAG-BASED)
            # -------------------------------------------------
            print("\n[STEP 3] Recommendation Generation (RAG + Fine-tuned Llama)")
            recommendation_result = recommendation_service.generate_recommendation(
                symptom=user_input,
                cause_or_disease=cause_explanation,
                top_k_docs=2,
                max_new_tokens=100,
            )

            recommendation = recommendation_result["recommendation"]
            print("Generated recommendation:")
            print(recommendation)

            # (Optional) Show retrieved treatment guidelines
            print("\nRetrieved treatment guidelines:")
            for i, doc in enumerate(recommendation_result["retrieved_guidelines"], 1):
                print(f"  {i}. {doc['disease']} (similarity: {doc['similarity']:.3f})")

            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("You can try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    main()
