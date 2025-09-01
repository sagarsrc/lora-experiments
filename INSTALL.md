# Installation Instructions

## CUDA-Enabled Installation

This project requires PyTorch with CUDA support. Follow these steps:

### 1. Install PyTorch with CUDA 12.1
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install PEFT from GitHub (latest features)
```bash
uv pip install git+https://github.com/huggingface/peft
```

### 3. Install remaining dependencies
```bash
uv pip install transformers matplotlib pandas tensorboard tqdm numpy seaborn jupyter
```

### 4. Or install all at once
```bash
# Install PyTorch with CUDA first
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install everything else
uv pip install git+https://github.com/huggingface/peft transformers matplotlib pandas tensorboard tqdm numpy seaborn jupyter
```

### Verify Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

## CPU-Only Installation
If you don't have CUDA, use:
```bash
uv sync  # This will install CPU versions
```