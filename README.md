# Ephemeral GPU VM Inference (Voltage Park)


## 0) Prereqs

- **Voltage Park VM** with NVIDIA GPU (e.g., H100, A100), **Ubuntu 22.04**.
- **SSH key** added at VM creation.

---

## 1) Launch & SSH


From the VM:
```bash
git clone https://github.com/amaders/WOA-VM.git
cd WOA-VM
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

Verify GPU Usage:

```bash
nvidia-smi
```
