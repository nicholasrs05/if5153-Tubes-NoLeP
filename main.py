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
    
    recommendation_service = RecommendationRAGService(
        rag_index_path="models/recommendation_rag_index.joblib",
        base_model_name="unsloth/Llama-3.2-1B-Instruct",
        adapter_path="models/llama-recommendation-fine-tuned",
        device="cpu",
    )

    print("Models loaded! You can now start the pipeline.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("\n\nEnter patient complaint: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the program.")
                break

            if not user_input:
                print("Empty input, please type a complaint sentence.\n")
                continue

            # -------------------------------------------------
            # 1) CLASSIFICATION MODEL
            # -------------------------------------------------
            print("\n[MODEL 1] Classification")
            predicted_label = classify_single_text(
                user_input, model, tokenizer, id2label, device
            )
            print(f"Predicted label: {predicted_label}")

            # -------------------------------------------------
            # 2) CAUSE GENERATION MODEL (RAG-BASED)
            # -------------------------------------------------
            print("\n[MODEL 2] Cause Generation (RAG + Qwen)")
            cause_result = cause_generation_service.generate_cause_explanation(
                complaint_text=user_input,
                predicted_category=predicted_label,
                top_k_docs=2,
                max_new_tokens=80,
            )

            cause_explanation = cause_result["cause_explanation"]
            print("Generated cause explanation:")
            print(cause_explanation)

            # print("\nRetrieved cause documents:")
            # for doc in cause_result["retrieved_docs"]:
            #     print(f"- ({doc['category']}, sim={doc['similarity']:.3f}) {doc['cause_text']}")

            # -------------------------------------------------
            # 3) RECOMMENDATION GENERATION MODEL (RAG-BASED)
            # -------------------------------------------------
            print("\n[MODEL 3] Recommendation Generation (RAG + Fine-tuned Llama)")
            recommendation_result = recommendation_service.generate_recommendation(
                symptom=user_input,
                cause_or_disease=cause_explanation,
                top_k_docs=2,
                max_new_tokens=80,
                use_rag_in_prompt=True,
                predicted_category=predicted_label,
            )

            recommendation = recommendation_result["recommendation"]
            print("Generated recommendation:")
            print(recommendation)

            # print("\nRetrieved treatment guidelines:")
            # for doc in recommendation_result["retrieved_guidelines"]:
            #     print(f"- ({doc['disease']}, sim={doc['similarity']:.3f}) {doc['treatment'][:150]}...")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("You can try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    main()
