import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model_name = "unsloth/Llama-3.2-1B-Instruct"
adapter_path = "https://drive.google.com/file/d/1uwfrCAn6k-GUPod5z3xnWXn6_-JVBSTW/view?usp=sharing"

# Load on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map=device,
    low_cpu_mem_usage=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()

# Inference function
def generate_recommendation(symptom, cause_or_disease):
    alpaca_prompt = """Based on the patient's symptoms and cause or disease mentioned, give a short recommendation to the patient (2 sentences max).

### Symptoms:
{}

### Cause or Disease:
{}

### Recommendation:
"""
    
    prompt = alpaca_prompt.format(symptom, cause_or_disease)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the recommendation part
    if "### Recommendation:" in response:
        answer = response.split("### Recommendation:")[-1].strip()
    else:
        answer = response.strip()
    
    return answer

# Test
symptom = "My lower back has been aching constantly and gets worse when I sit for long periods."
cause_or_disease = "The symptoms indicate lower back pain, likely caused by poor posture or muscle strain."

recommendation = generate_recommendation(symptom, cause_or_disease)
print(f"Symptom: {symptom}")
print(f"Cause: {cause_or_disease}")
print(f"Recommendation: {recommendation}")