from classification.predictions import classify_single_text, load_model

if __name__ == "__main__":
    print("Loading model... please wait.")
    model_classification = "models/roberta_classification"
    # model_similarity = "models/similarity_model.joblib"
    # model_seq2seq = "models/seq2seq_model"
    
    model, tokenizer, id2label, device = load_model(model_classification)
    print("Model loaded! You can now start classifying text.")
    while True:
        try:
            prompt = input("Enter your prompt: ")

            if prompt.lower() in ['exit', 'quit']:
                print("Exiting the program.")
                break

            elif prompt.lower().startswith('classify:'):
                text = prompt.replace("classify:", "").strip()
                prediction = classify_single_text(text, model, tokenizer, id2label, device)
                print(f"Predicted label: {prediction}")

            elif prompt.lower().startswith('similarity:'):
                pass
                # text = prompt.replace("similarity:", "").strip()
                # prediction = classify_single_text(text, model, tokenizer, id2label, device)
                # print(f"Predicted label: {prediction}")
            elif prompt.lower().startswith('seq2seq:'):
                pass
                # text = prompt.replace("similarity:", "").strip()
                # prediction = classify_single_text(text, model, tokenizer, id2label, device)
                # print(f"Predicted label: {prediction}")
        

            
            else:
                pass
                # Classification logic

                # Comparison logic

                # Seq2seq logic

        except Exception as e:
            print(f"An error occurred: {e}")