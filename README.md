# Tugas Besar IF5153 - Pemrosesan Bahasa Alami

## Group 17

| NIM | Name |
| --- | ---- |
| 13522144 | Nicholas Reymond Sihite |
| 13522151 | Samy Muhammad Haikal |
| 13522159 | Rafif Ardhinto Ichwantoro |

## System Overview

This is an AI-based medical consultation system that analyzes patient complaints and provides diagnosis and treatment recommendations. The system uses 3 NLP models in a pipeline:

### Model 1: Classification
- **Purpose**: Categorize patient complaints into medical categories
- **Model**: Fine-tuned RoBERTa

### Model 2: Cause Generation (RAG)
- **Purpose**: Explain possible causes and diseases
- **Model**: Retrieval-Augmented Generation with SentenceTransformers + Qwen

### Model 3: Recommendation Generation (RAG)
- **Purpose**: Provide personalized treatment recommendations
- **Model**: RAG with SentenceTransformers + Fine-tuned Llama 3.2-1B (LoRA)

### Workflow
1. **Patient enters complaint** → Model 1 classifies it
2. **Classification + Complaint** → Model 2 generates cause/disease explanation
3. **Complaint + Cause** → Model 3 retrieves guidelines and generates recommendation

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the System

```bash
python main.py
```
Then enter patient complaints when prompted.


### Example Usage
```
Enter patient complaint: My lower back hurts when I sit for long periods

[STEP 1] Classification
Predicted label: Back pain

[STEP 2] Cause Generation (RAG + Qwen)
Generated cause explanation:
The symptoms indicate lower back pain, likely caused by poor posture or muscle strain.

[STEP 3] Recommendation Generation (RAG + Fine-tuned Llama)
Generated recommendation:
Consider improving your posture and taking regular breaks. Apply heat therapy and gentle stretching exercises.
```

---