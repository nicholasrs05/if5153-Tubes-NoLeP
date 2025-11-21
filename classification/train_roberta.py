import argparse
import json
import os
from datasets import Dataset
import evaluate
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np


def compute_metrics(pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


def main(args):
    # load data
    df = pd.read_csv(args.data)

    if "Patient_comment" not in df.columns or "Patient_Category" not in df.columns:
        raise ValueError("CSV must contain Patient_comment and Patient_Category columns")

    df = df[["Patient_comment", "Patient_Category"]].dropna().reset_index(drop=True)

    # label encode
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Patient_Category"])

    # save label mapping
    os.makedirs(args.output_dir, exist_ok=True)
    label2id = {label: int(idx) for idx, label in enumerate(le.classes_)}
    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    # split
    train_frac = args.train_frac
    df_shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_end = int(len(df_shuffled) * train_frac)
    df_train = df_shuffled.iloc[:train_end]
    df_eval = df_shuffled.iloc[train_end:]

    dataset_train = Dataset.from_pandas(df_train[["Patient_comment", "label"]])
    dataset_eval = Dataset.from_pandas(df_eval[["Patient_comment", "label"]])

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_fn(ex):
        return tokenizer(ex["Patient_comment"], truncation=True)

    dataset_train = dataset_train.map(lambda x: tokenizer(x["Patient_comment"], truncation=True, padding=False), batched=True)
    dataset_eval = dataset_eval.map(lambda x: tokenizer(x["Patient_comment"], truncation=True, padding=False), batched=True)

    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
      output_dir=args.output_dir,
      do_eval=True,
      eval_steps=500,
      save_steps=500,
      save_total_limit=2,
      learning_rate=args.learning_rate,
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      num_train_epochs=args.epochs,
      weight_decay=0.01,
      logging_dir=os.path.join(args.output_dir, "logs"),
      logging_steps=50,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    print("Training finished. Model and tokenizer saved to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for patient complaint classification")
    parser.add_argument("--data", type=str, default="data/HealthCare Data.csv", help="Path to CSV data")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="models/roberta_classification", help="Where to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
