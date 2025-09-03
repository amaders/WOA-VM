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




## Setting Up and Launching VM

1. `ssh-keygen` in the terminal and give it a name, e.g. “voltage” (don't necessarily need to add password)
2. Upload the public ssh key to voltage park - organisation - managing ssh keys
3. Create the VM via voltage park website
4. From Cursor/VS Code, click "Connect Via SSH" -> "Configure SSH Hosts..." -> ".ssh/config" (If it exists, otherwise need to create .ssh file) -> copy and paste

```bash
Host voltage
    HostName [host name, something like xxx.xx.xx.xxx]
    Port [port name, following the -p]
    User user
    IdentityFile ~/.ssh/[ssh key name here]
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes   # macOS
```

5. Click out of the file, click on "Connect Via SSH", select "voltage"

## Configuring VM

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
