Quick guide — Fine-tune RoBERTa for patient complaint classification

Prerequisites
- Python 3.8+
- Create and activate a virtual environment, then install dependencies:

PowerShell example:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Train

Run training with defaults (uses `data/HealthCare Data.csv` and `roberta-base`):

```powershell
python classification\train_roberta.py --output_dir models\roberta_classification
```

Common options
- `--data`: path to CSV (default `data/HealthCare Data.csv`)
- `--model_name`: Hugging Face model name (default `roberta-base`)
- `--epochs`: number of epochs (default 3)
- `--batch_size`: batch size per device (default 16)
- `--train_frac`: fraction of data used for training (default 0.9)

Outputs
- Fine-tuned model and tokenizer saved under the `--output_dir` path.
- Label mapping saved as `label2id.json` in the same folder.

Notes
- For small dataset sizes consider lowering batch size or using gradient accumulation.
- To evaluate or serve the model load it with `AutoModelForSequenceClassification.from_pretrained(<output_dir>)`.
