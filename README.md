# LoRA Experiments

A comprehensive comparison of **baseline fine-tuning** vs **LoRA (Low-Rank Adaptation)** for sentiment analysis using Llama 3.2 1B on the IMDB dataset.

## What This Repository Does

This repository demonstrates the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) using LoRA compared to traditional full fine-tuning approaches. It includes:

- **Baseline Implementation**: Traditional fine-tuning of Llama 3.2 1B with frozen early layers
- **LoRA Implementation**: Parameter-efficient fine-tuning using Low-Rank Adaptation 
- **Comprehensive Evaluation**: Detailed metrics comparison including F1, precision, recall, and confusion matrices
- **Automated Analysis**: Side-by-side comparison of parameter efficiency vs performance trade-offs

## Repository Structure

```
├── notebooks/
│   ├── 001-baseline_llama.py         # Partial fine-tuning (12 layers frozen)
│   ├── 002-lora_llama.py             # LoRA fine-tuning implementation  
│   ├── 003-compare_results.py        # Automated comparison script
│   ├── 004-full-vs-lora-batch-demo.py # True full fine-tuning vs LoRA batch size demo
│   ├── .env.example                  # Environment setup template
│   ├── results_baseline_*.json       # Baseline training results
│   ├── results_lora_*.json           # LoRA training results
│   └── batch_comparison_*.json       # Batch size comparison results
├── pyproject.toml                    # Project dependencies
├── README.md                         # This file
└── INSTALL.md                        # Installation instructions
```

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.1 compatible GPU
- uv package manager

### Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd lora-experiments
   ```

2. **Initialize Project with uv**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   uv sync
   ```

4. **Setup Environment**:
   ```bash
   cp notebooks/.env.example notebooks/.env
   # Edit notebooks/.env and add your HuggingFace token:
   # HF_TOKEN=hf_your_token_here
   ```

5. **Verify Installation**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Quick Start

```bash
# Run partial fine-tuning (baseline)
python notebooks/001-baseline_llama.py

# Run LoRA fine-tuning  
python notebooks/002-lora_llama.py

# Compare results
python notebooks/003-compare_results.py

# Demonstrate true full fine-tuning vs LoRA batch size differences
python notebooks/004-full-vs-lora-batch-demo.py
```

## Key Findings

### Performance Comparison
- **Baseline**: 87.16% accuracy with 505M trainable parameters (40.9% of model)
- **LoRA**: 93.84% accuracy with 11M trainable parameters (0.9% of model)
- **Improvement**: +6.68% accuracy with 97.8% fewer trainable parameters

**NOTE**: The "baseline" is not true full fine-tuning - we freeze the first 12 layers (out of 22) to make training feasible on consumer GPUs with batch size 16. Full fine-tuning of all 1.24B parameters would require significantly more GPU memory and compute resources. This partial fine-tuning approach gives us a practical comparison point while demonstrating the computational constraints that make LoRA attractive.

### Parameter Efficiency
| Method | Total Params | Trainable Params | Trainable % | Accuracy | Notes |
|--------|-------------|------------------|-------------|----------|-------|
| Baseline* | 1.24B | 506M | 40.9% | 87.16% | *Partial fine-tuning (12 layers frozen) |
| LoRA | 1.24B | 11M | 0.9% | 93.84% | Full model + adapters |

## GPU Memory Usage Analysis

**Training vs Inference Memory Usage:**

**During Training:**
1. **Full Fine-tuning**: Requires massive GPU memory for gradients of all trainable parameters
   - Even batch size 1 may not fit on consumer GPUs for 1B+ models
   - All layer gradients must be stored and updated
   - Memory scales with number of trainable parameters
   - **Note**: Our demo uses `device_map="auto"` and `torch.float16` to attempt full fine-tuning (requires `accelerate` package)

2. **LoRA Training**: Dramatically reduces memory requirements
   - Can use much larger batch sizes (16-32+) 
   - Only stores gradients for small adapter matrices (~11M parameters)
   - Base model weights frozen - no gradient storage needed for them
   - **Memory savings are substantial during training**

**During Inference:**
- Both approaches load the full model into memory
- LoRA adds minimal overhead (~11M adapter parameters)
- Memory usage is similar for inference only

3. **Why Our Baseline Uses Similar Memory**:
   - Our "baseline" freezes 12/22 layers, reducing trainable parameters to 40.9%
   - This is NOT true full fine-tuning but a memory optimization
   - True full fine-tuning would require much more memory and smaller batches

**LoRA's Real Benefits**:
- **Training Memory Efficiency**: Massive reduction in GPU memory during training
- **Larger Batch Sizes**: Can use 8-16x larger batches than full fine-tuning
- **Parameter Efficiency**: 97.8% reduction in trainable parameters (505M to 11M)
- **Storage Efficiency**: Save only LoRA weights (~MB) vs full model (~GB)
- **Better Performance**: Often achieves better results due to regularization effect
- **Faster Training**: Higher batch sizes + fewer parameters = faster convergence

**Key Insight**: LoRA's primary advantage is **training memory efficiency**. While inference memory is similar, training memory is dramatically reduced, enabling larger batch sizes and more efficient training on consumer GPUs.

## Technical Details

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32  
- **Dropout**: 0.1
- **Target Modules**: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### Training Configuration
- **Dataset**: IMDB sentiment analysis (5K train samples, 2.5K test)
- **Model**: meta-llama/Llama-3.2-1B
- **Batch Size**: 16 (enabled by layer freezing in baseline)
- **Max Length**: 256 tokens
- **Learning Rate**: 2e-5 (baseline), 2e-4 (LoRA)
- **Epochs**: 1 with early stopping
- **Memory Optimization**: Baseline freezes first 12/22 layers; LoRA trains all layers via adapters

## Dependencies

- PyTorch 2.5.1+ with CUDA 12.1 support
- Transformers 4.56.0+
- PEFT (latest from GitHub)
- Accelerate 1.0.0+ (required for `device_map="auto"`)
- scikit-learn for evaluation metrics
- Datasets for IMDB loading

## Comparison Results

Running `python compare_results.py` produces this comprehensive analysis:

```
Loading results:
  Baseline: results_baseline_1756755949.json
  LoRA: results_lora_1756756758.json

================================================================================
LLAMA 3.2 1B SENTIMENT ANALYSIS: BASELINE vs LoRA COMPARISON
================================================================================

PERFORMANCE METRICS
--------------------------------------------------
  Accuracy     | Baseline: 0.8716 | LoRA: 0.9384 | Δ: +0.0668
  F1_Score     | Baseline: 0.8710 | LoRA: 0.9384 | Δ: +0.0673
  Precision    | Baseline: 0.8776 | LoRA: 0.9391 | Δ: +0.0615
  Recall       | Baseline: 0.8716 | LoRA: 0.9384 | Δ: +0.0668
  Loss         | Baseline: 0.4278 | LoRA: 0.1939 | Δ: +0.2339

PARAMETER EFFICIENCY
--------------------------------------------------
  Total Parameters:
    Baseline: 1,235,818,498
    LoRA:     1,247,090,690

  Trainable Parameters:
    Baseline: 505,960,450 (40.9%)
    LoRA:     11,276,290 (0.9%)

  Efficiency Gains:
    Parameter Reduction: 97.8%
    Efficiency Ratio: 44.9x fewer trainable parameters
    Accuracy per Million Params:
      Baseline: 0.00
      LoRA:     0.08 (48.3x better)

PER-CLASS PERFORMANCE
--------------------------------------------------
  Negative:
    Precision | Baseline: 0.925 | LoRA: 0.956 | Δ: +0.031
    Recall    | Baseline: 0.807 | LoRA: 0.918 | Δ: +0.111
    F1-Score  | Baseline: 0.862 | LoRA: 0.937 | Δ: +0.075
  Positive:
    Precision | Baseline: 0.831 | LoRA: 0.922 | Δ: +0.091
    Recall    | Baseline: 0.936 | LoRA: 0.959 | Δ: +0.023
    F1-Score  | Baseline: 0.880 | LoRA: 0.940 | Δ: +0.060

CONFUSION MATRICES
--------------------------------------------------
  Baseline:
    Predicted:  Neg   Pos
    Actual Neg: 1003   240
    Actual Pos:  81   1176
    Accuracy: 0.872

  LoRA:
    Predicted:  Neg   Pos
    Actual Neg: 1141   102
    Actual Pos:  52   1205
    Accuracy: 0.938

OVERALL ASSESSMENT
--------------------------------------------------
  LoRA vs Baseline:
    Parameter Efficiency: EXCELLENT (97.8% reduction)
    Performance: SUPERIOR (+0.067 accuracy)

  Verdict: LoRA is highly effective! Major parameter savings with minimal performance impact.

================================================================================
```

## Additional Resources

See `outputs.md` for detailed training logs and raw experimental results.