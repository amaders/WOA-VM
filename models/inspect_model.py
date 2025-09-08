import torch
from nnsight import NNsight, LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"  # or the base model

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  
    device_map="auto" #automatically stores model
).requires_grad_(False).eval()

nn_model = NNsight(model)

print(nn_model)

#this is an instruction-tuned model, so we need to specify it in chat format
prompt = "I really like nuclear power because"

# 2) Tokenize to get BOTH input_ids and attention_mask
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    # Single forward pass to get logits
    outputs = model.forward(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    
    # Extract logits from the output
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Last token logits shape: {logits[:, -1, :].shape}")  # Next token predictions
    
    # Decoding step: get the most likely next token
    next_token_logits = logits[:, -1, :]  # Get logits for the last position
    next_token_id = torch.argmax(next_token_logits, dim=-1)  # Get most likely token ID
    
    # Decode the token ID back to text
    next_token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
    
    print(f"Next token ID: {next_token_id.item()}")
    print(f"Next token text: '{next_token_text}'")
    
    # Optional: show top-k predictions
    top_k = 5
    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
    print(f"\nTop {top_k} next token predictions:")
    for i in range(top_k):
        token_id = top_k_indices[0, i].item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        logit_value = top_k_logits[0, i].item()
        print(f"  {i+1}. Token ID {token_id}: '{token_text}' (logit: {logit_value:.2f})")




