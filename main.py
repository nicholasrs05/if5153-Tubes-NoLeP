from classification.predictions import classify_single_text, load_model
from cause_textgen.cause_generation_service import CauseRAGService



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

    # ---- Placeholder seq2seq (belum dipakai) ----
    # seq2seq_ready = False
    # seq2seq_model = None
    # if seq2seq_ready:
    #     seq2seq_model = Seq2SeqRecommendationModel(model_path="path/to/your/seq2seq/model")

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
            # 3) SEQ2SEQ RECOMMENDATION STEP (BELUM AKTIF)
            # -------------------------------------------------
            # print("\n[STEP 3] Recommendation (Seq2Seq)")
            # if seq2seq_ready and seq2seq_model is not None:
            #     recommendation = seq2seq_model.generate_recommendation(
            #         complaint_text=user_input,
            #         predicted_category=predicted_label,
            #         cause_explanation=cause_explanation,
            #     )
            #     print("Generated recommendation:")
            #     print(recommendation)
            # else:
            #     print("Seq2Seq model not ready yet. This step will be enabled later.")

            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("You can try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    main()
