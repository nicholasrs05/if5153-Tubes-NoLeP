import argparse
import json
import os
from typing import List

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
from tqdm import tqdm
import gdown

def download_model(model_dir, folder_id):
    if not os.path.exists(model_dir):
        print(f"Model not found. Downloading from Google Drive...")
        os.makedirs(model_dir, exist_ok=True)
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=model_dir, quiet=False, use_cookies=False)
    else:
        print(f"Model found locally: {model_dir}")

def read_csv_with_encodings(path, encodings=None):
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", engine="python", errors="replace")


def batch_iter(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main(args):
    # load data
    download_model(args.model_dir, folder_id="1GlKw76LOp7KuoGy_t6l-tuNLktFxC9fR")
    df = read_csv_with_encodings(args.data)
    if "Patient_comment" not in df.columns or "Patient_Category" not in df.columns:
        raise ValueError("CSV must contain Patient_comment and Patient_Category columns")

    texts = df["Patient_comment"].fillna("").astype(str).tolist()
    actual_labels = df["Patient_Category"].fillna("").astype(str).tolist()

    # load label mapping
    label_path = os.path.join(args.model_dir, "label2id.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"label2id.json not found in {args.model_dir}")
    with open(label_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    # ensure ints
    label2id = {k: int(v) for k, v in label2id.items()}
    id2label = {int(v): k for k, v in label2id.items()}

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds_labels = []
    with torch.no_grad():
        for batch_texts in tqdm(list(batch_iter(texts, args.batch_size)), desc="Predicting"):
            enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_pred_labels = [id2label.get(int(p), "UNKNOWN") for p in batch_preds]
            preds_labels.extend(batch_pred_labels)
    print("\n=== Sample Predictions (Patient_comment | Predicted | Actual) ===")
    for comment, pred, actual in zip(texts[:10], preds_labels[:10], actual_labels[:10]):
      print(f"Comment: {comment}")
      print(f"Predicted: {pred} | Actual: {actual}\n")
    # map string labels to ids for evaluation; filter out any rows with unknown mappings
    preds_ids = []
    actual_ids = []
    skipped = 0
    for p_str, a_str in zip(preds_labels, actual_labels):
        p_id = label2id.get(p_str)
        a_id = label2id.get(a_str)
        if p_id is None or a_id is None:
            skipped += 1
            continue
        preds_ids.append(p_id)
        actual_ids.append(a_id)

    if len(preds_ids) == 0:
        print("No valid label mappings found between predictions and label2id.json. Cannot compute metrics.")
    else:
        metric_acc = evaluate.load("accuracy")
        metric_f1 = evaluate.load("f1")
        acc = metric_acc.compute(predictions=preds_ids, references=actual_ids)["accuracy"]
        f1 = metric_f1.compute(predictions=preds_ids, references=actual_ids, average="macro")["f1"]

        print(f"Accuracy: {acc:.4f} (computed on {len(preds_ids)} samples, skipped {skipped})")
        print(f"F1 (macro): {f1:.4f}")

    # save predictions (keep original string categories for readability)
    out_df = pd.DataFrame({
        "Patient_comment": texts,
        "Predicted_Category": preds_labels,
        "Actual_Category": actual_labels,
    })
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print("Predictions saved to:", args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/roberta_classification")
    parser.add_argument("--data", type=str, default="data/HealthCare Data.csv")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)