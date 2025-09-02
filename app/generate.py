import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
PROMPT = os.environ.get("PROMPT", "Say hello in one sentence.")
DTYPE  = os.environ.get("DTYPE", "auto")   # "auto"|"bfloat16"|"float16"|"float32"
TEMP   = float(os.environ.get("TEMP", "1"))
TOP_P  = float(os.environ.get("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
USE_FLASH = os.environ.get("USE_FLASH", "1") == "1"  # best-effort

torch.backends.cuda.matmul.allow_tf32 = True

print(f"[generate] loading tokenizer for {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

attn_impl = "flash_attention_2" if USE_FLASH else None
print(f"[generate] loading model {MODEL} (dtype={DTYPE}, flash={USE_FLASH}) ...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE if DTYPE != "auto" else "auto",
    device_map="auto",
    attn_implementation=attn_impl,  # silently ignored if not available
)
print(f"[generate] model loaded in {time.time()-t0:.1f}s")

inputs = tok(PROMPT, return_tensors="pt").to(model.device)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

gen_kwargs = dict(
    do_sample=True,
    temperature=TEMP,
    top_p=TOP_P,
    max_new_tokens=MAX_NEW_TOKENS,
    streamer=streamer,
)

print("[generate] generating...\n")
with torch.inference_mode():
  _ = model.generate(**inputs, **gen_kwargs)
print("\n[generate] done.")