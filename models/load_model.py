import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-14B"  # or the base model

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  
    device_map="auto" #automatically stores model
).eval()

print(model)

# Simple text prompt for generation testing (not instruction-tuned)
prompt = "I really like nuclear power because"

# Tokenize the prompt directly
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # ✓ auto-made mask
        max_new_tokens=256,
    )

# Decode only the new tokens (skip the prompt part)
generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(generated_text)
