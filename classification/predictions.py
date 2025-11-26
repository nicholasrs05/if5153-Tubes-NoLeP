import json
import os
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

def classify_single_text(text, model, tokenizer, id2label, device):
    enc = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        pred = torch.argmax(logits, dim=-1).item()

    return id2label.get(pred, "UNKNOWN")

def load_model(model_dir):
    download_model(model_dir, folder_id="1GlKw76LOp7KuoGy_t6l-tuNLktFxC9fR")

    # Load label mapping
    label_path = os.path.join(model_dir, "label2id.json")
    with open(label_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, id2label, device