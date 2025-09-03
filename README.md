# Ephemeral GPU VM Inference (Voltage Park) — One-Page README

Run **transformers-only** inference on a throwaway GPU VM with **uv** (no Docker, no S3). This guide goes from clean Ubuntu VM → env setup → model download → inference → shutdown.

---

## Repo Structure

```bash
ml-ephemeral/
├─ app/
│  └─ generate.py            # one-prompt inference (transformers)
├─ scripts/
│  └─ bootstrap_uv.sh        # env setup, Torch CUDA wheel, HF cache config
├─ pyproject.toml            # project deps (managed by uv)
├─ .gitignore
├─ .env.example              # (optional) HF_TOKEN placeholder
└─ README.md                 # this file
```

## 0) Prereqs

- **Voltage Park VM** with NVIDIA GPU (e.g., H100, A100), **Ubuntu 22.04**.
- **SSH key** added at VM creation.
- (Optional) **Hugging Face token** for gated/private models (e.g., Llama 3.x).

---

## 1) Launch & SSH

From your laptop:
```bash
ssh ubuntu@<VM_PUBLIC_IP>
```

From the VM:
```bash
git clone https://github.com/amaders/WOA-VM.git
cd ml-ephemeral
```

Bootstrap the environment:

```bash
chmod +x scripts/bootstrap_uv.sh
./scripts/bootstrap_uv.sh
```

Load environmental variables into current shell:

```bash
source ~/.bashrc
```

(if you want to use a gated/private model)

```bash
export HF_TOKEN=hf_xxx
```

Run inference:

```bash
MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
PROMPT="Explain tensor parallelism in 2 sentences." \
MAX_NEW_TOKENS=100 TEMP=0.5 DTYPE=bfloat16 USE_FLASH=1 \
uv run python app/generate.py
```

Verify GPU Usage:

```bash
nvidia-smi
```

Stop the VM:

```bash
sudo poweroff
```
