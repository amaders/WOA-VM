import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"  # or the base model

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  
    device_map="auto" #automatically stores model
).eval()

print(model)

#this is an instruction-tuned model, so we need to specify it in chat format
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello! Can you tell me a fun fact about space?"}
]

chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,                 # <-- get a string
    add_generation_prompt=True
)

# 2) Tokenize to get BOTH input_ids and attention_mask
inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # ✓ auto-made mask
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

# Decode only the new tokens (skip the prompt part)
generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(generated_text)
