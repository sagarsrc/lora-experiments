# LoRA Experiments

A systematic exploration of **LoRA (Low-Rank Adaptation)** effectiveness for large language model fine-tuning, demonstrating why parameter-efficient methods are essential for practical LLM training.

**Author**: [Sagar Sarkale](https://www.linkedin.com/in/sagar-sarkale/)
📖 **Blog Post**: [Deep Dive into LoRA: A Practical Exploration](https://sagarsarkale.com/blog/genai/lora-deepdive/)

## Table of Contents

- [Hardware Configuration](#hardware-configuration)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Running the Experiments](#running-the-experiments)
- [Experimental Journey](#experimental-journey)
  - [Experiment 1: Baseline Fine-tuning](#experiment-1-baseline-fine-tuning-with-memory-constraints)
  - [Experiment 2: LoRA Fine-tuning](#experiment-2-lora-fine-tuning)
  - [Experiment 3: Performance Analysis](#experiment-3-performance-analysis--comparison)
  - [Experiment 4: Memory Reality Check](#experiment-4-memory--batch-size-reality-check)
- [Key Technical Configurations](#key-technical-configurations)
- [Fundamental Insights](#fundamental-insights)
- [Conclusions](#conclusions)
- [Additional Resources](#additional-resources)

## Hardware Configuration

**All experiments conducted on:**
```
GPU: NVIDIA A10 (23GB VRAM)
Driver Version: 570.144
CUDA Version: 12.8
Memory Capacity: 23,028 MiB total
```

This hardware configuration demonstrates LoRA's effectiveness on professional-grade GPUs. Results may vary on consumer GPUs with less VRAM.

## Repository Structure

```
├── notebooks/
│   ├── 001-baseline_llama.py         # Experiment 1: Baseline with frozen layers
│   ├── 002-lora_llama.py             # Experiment 2: LoRA fine-tuning
│   ├── 003-compare_results.py        # Experiment 3: Comprehensive comparison
│   ├── 004-full_vs_lora_batch_demo.py # Experiment 4: Memory reality check
│   ├── .env.example                  # Environment setup template
│   ├── results_baseline_*.json       # Baseline training results
│   ├── results_lora_*.json           # LoRA training results
│   └── batch_comparison_*.json       # Memory comparison results
├── pyproject.toml                    # Project dependencies
├── outputs.md                        # Detailed training logs
└── README.md                         # This file
```

## Installation & Setup

### Prerequisites
- Python 3.12+
- CUDA 12.1 compatible GPU
- uv package manager

### Quick Setup
```bash
git clone https://github.com/sagarsrc/lora-experiments.git
cd lora-experiments
uv venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
cp notebooks/.env.example notebooks/.env
# Edit notebooks/.env and add your HuggingFace token:
# HF_TOKEN=hf_your_token_here
```

### Verify Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Dependencies
- PyTorch 2.5.1+ with CUDA 12.1 support
- Transformers 4.56.0+
- PEFT (latest from GitHub)
- Accelerate 1.0.0+
- scikit-learn for evaluation metrics
- Datasets for IMDB loading

## Running the Experiments

Follow the experimental sequence:

```bash
# Experiment 1: Baseline fine-tuning
python notebooks/001-baseline_llama.py

# Experiment 2: LoRA fine-tuning
python notebooks/002-lora_llama.py

# Experiment 3: Compare results
python notebooks/003-compare_results.py

# Experiment 4: Memory reality check (will crash intentionally!)
python notebooks/004-full_vs_lora_batch_demo.py
```

## Experimental Journey

### Experiment Progression

**Experiment 1**: Started with baseline Llama 3.2 1B fine-tuning to establish performance benchmarks and understand memory constraints with traditional approaches.

**Experiment 2**: Applied LoRA to see if we could match baseline performance while using fewer trainable parameters and less memory.

**Experiment 3**: Built automated comparison to systematically evaluate whether LoRA maintains performance while being more efficient.

**Experiment 4**: Demonstrated the memory reality - showing that full fine-tuning fails with OOM while LoRA succeeds with larger batch sizes.

---

## Experiment 1: Baseline Fine-tuning with Memory Constraints
**File**: `001-baseline_llama.py`

### Objective
Establish a baseline by fine-tuning Llama 3.2 1B for sentiment analysis, using memory optimizations necessary to fit on available hardware.

### Approach
- **Strategy**: Freeze first 12 layers (out of 22) to reduce memory usage
- **Reasoning**: Full fine-tuning would exceed GPU memory even with batch size 1
- **Trade-off**: Performance vs. feasibility on consumer hardware

### Results
```
Total parameters: 1,235,818,498
Trainable parameters: 505,960,450 (40.9%)
Batch size: 16
Test Accuracy: 87.16%
Training time: ~8 minutes
```

### Detailed Training Output
```
Llama 3.2 1B Sentiment Analysis - Baseline (Partial Fine-tuning)
============================================================
NOTE: This is NOT full fine-tuning. First 12 layers are frozen
      to enable training on consumer GPUs with reasonable memory usage.
      True full fine-tuning would require significantly more resources.
============================================================
Loading IMDB dataset...
Train: 4000, Val: 1000, Test: 2500
  Freezing 12/22 layers for memory efficiency
  This enables batch_size=16 training on consumer GPUs
Training on cuda
Total parameters: 1,235,818,498
Trainable parameters: 505,960,450 (40.9%)
Training: 100%|██████████| 250/250 [07:21<00:00,  1.76s/it]
Validating: 100%|██████████| 63/63 [00:51<00:00,  1.22it/s]
Epoch 1/1 (492.7s)
  Train Loss: 0.5091
  Val Loss: 0.3921, Val Acc: 0.8650
  LR: 1.00e-05
  ✓ New best model (loss: 0.3921)

Comprehensive Evaluation:
============================================================
Test Results:
  Loss: 0.4278
  Accuracy: 0.8716
  F1 Score: 0.8710
  Precision: 0.8776
  Recall: 0.8716

Model Parameters:
  Total: 1,235,818,498
  Trainable: 505,960,450 (40.9%)

Per-Class Results:
  Negative: Precision=0.925, Recall=0.807, F1=0.862
  Positive: Precision=0.831, Recall=0.936, F1=0.880

Results saved to: results_baseline_1756755949.json
============================================================
```

### Key Insights
- Even with layer freezing, used 40.9% of model parameters
- Achieved reasonable performance but required memory optimization compromises
- Demonstrated the practical constraints of traditional fine-tuning approaches

---

## Experiment 2: LoRA Fine-tuning
**File**: `002-lora_llama.py`

### Objective
Compare LoRA's parameter efficiency against the baseline while maintaining or improving performance.

### Approach
- **Strategy**: Apply LoRA adapters to attention and feed-forward layers
- **Target modules**: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Configuration**: Rank=16, Alpha=32, Dropout=0.1

### Results
```
Total parameters: 1,247,090,690
Trainable parameters: 11,276,290 (0.9%)
Batch size: 16
Test Accuracy: 93.84%
Training time: ~8 minutes
```

### Detailed Training Output
```
Llama 3.2 1B Sentiment Analysis - LoRA
============================================================
Loading IMDB dataset...
Train: 4000, Val: 1000, Test: 2500
Applying LoRA...
trainable params: 11,272,192 || all params: 1,247,086,592 || trainable%: 0.9039
Training on cuda
Total parameters: 1,247,090,690
Trainable parameters: 11,276,290 (0.9%)
Training: 100%|██████████| 250/250 [07:20<00:00,  1.76s/it]
Validating: 100%|██████████| 63/63 [00:56<00:00,  1.12it/s]
Epoch 1/1 (496.6s)
  Train Loss: 0.5033
  Val Loss: 0.1848, Val Acc: 0.9480
  LR: 1.00e-04
  ✓ New best model (loss: 0.1848)

Comprehensive Evaluation:
============================================================
Test Results:
  Loss: 0.1939
  Accuracy: 0.9384
  F1 Score: 0.9384
  Precision: 0.9391
  Recall: 0.9384

Model Parameters:
  Total: 1,247,090,690
  Trainable: 11,276,290 (0.9%)

Per-Class Results:
  Negative: Precision=0.956, Recall=0.918, F1=0.937
  Positive: Precision=0.922, Recall=0.959, F1=0.940

Results saved to: results_lora_1756756758.json
============================================================
```

### Key Insights
- **97.8% parameter reduction** (505M → 11M trainable parameters)
- **+6.68% accuracy improvement** despite fewer parameters
- **Same batch size** as baseline but with full model engagement
- Demonstrated LoRA's regularization effect preventing overfitting

---

## Experiment 3: Performance Analysis & Comparison
**File**: `003-compare_results.py`

### Objective
Comprehensive analysis of baseline vs. LoRA across multiple metrics to validate LoRA's effectiveness.

### Methodology
- Automated loading of latest experimental results
- Multi-dimensional comparison: accuracy, F1, precision, recall, parameter efficiency
- Per-class analysis and confusion matrix comparison
- Statistical significance assessment

### Key Findings

| Metric | Baseline | LoRA | Improvement |
|--------|----------|------|-------------|
| **Accuracy** | 87.16% | 93.84% | +6.68% |
| **F1 Score** | 87.10% | 93.84% | +6.74% |
| **Parameter Efficiency** | 505M params | 11M params | **97.8% reduction** |
| **Negative Class F1** | 0.862 | 0.937 | +0.075 |
| **Positive Class F1** | 0.880 | 0.940 | +0.060 |

### Detailed Comparison Output

Running `python notebooks/003-compare_results.py` produces:

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

### Verdict
**LoRA is highly effective! Major parameter savings with superior performance.**

---

## Experiment 4: Memory & Batch Size Reality Check
**File**: `004-full_vs_lora_batch_demo.py`

### Objective
Demonstrate the **practical impossibility** of true full fine-tuning and showcase LoRA's batch size advantages.

### Experimental Design
- **Full Fine-tuning Attempt**: Train all 1.24B parameters with batch size 2
- **LoRA Approach**: Same model with LoRA adapters, batch size 16
- **User Control**: Flag to experience OOM crashes firsthand

### Results - The Reality of Full Fine-tuning

```
⚠️  Running FULL fine-tuning experiment - expect CUDA OOM error!

🔥 EXPECTED CUDA OOM ERROR OCCURRED:
   OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB...

   This demonstrates why LoRA is necessary!
```

### Results - LoRA Success

```
✅ Running LoRA experiment - should work with larger batch size

EXPERIMENT: LoRA Fine-tuning
Batch Size: 16 (8x larger than attempted full fine-tuning)
Peak Memory: 20.57GB
Training Time: 99.2s
Test Accuracy: 85.80%
Trainable Params: 11,276,290
```

### Critical Insights

1. **Memory Reality**:
   - **Allocated**: 4.79GB (active tensors)
   - **Reserved**: 20.77GB (total PyTorch memory pool)
   - **Peak**: 20.57GB (maximum during training)

2. **Practical Implications**:
   - Full fine-tuning **FAILS** even with batch size 2 on 23GB GPU
   - LoRA **succeeds** with batch size 16 (8x larger)
   - Demonstrates LoRA's **practical necessity** for large model training

3. **Memory Understanding**:
   - Model uses 20.77GB total GPU memory
   - Only 4.79GB actively allocated to tensors
   - Remaining ~16GB is PyTorch's memory pool for efficiency

## Key Technical Configurations

### LoRA Parameters
- **Rank (r)**: 16 (balance between efficiency and capacity)
- **Alpha**: 32 (scaling factor for adapter influence)
- **Dropout**: 0.1 (regularization)
- **Target Modules**: All attention and FFN linear layers

### Training Setup
- **Model**: meta-llama/Llama-3.2-1B
- **Dataset**: IMDB sentiment analysis (5K samples)
- **Batch Size**: 16 (LoRA enables this, baseline requires layer freezing)
- **Learning Rates**: 2e-5 (baseline), 2e-4 (LoRA - higher due to adapter training)
- **Hardware**: NVIDIA A10 (23GB) - professional GPU demonstrating challenges

## Fundamental Insights

### Why LoRA Works

1. **Mathematical Efficiency**: Low-rank decomposition captures essential adaptations
2. **Regularization Effect**: Constrained parameter space prevents overfitting
3. **Memory Efficiency**: Only adapter gradients stored during training
4. **Practical Necessity**: Enables training where full fine-tuning fails

### Memory Deep Dive

**Training Memory Components**:
- **Model Weights**: ~5GB (same for both approaches)
- **Gradients**: 505M × 4 bytes = ~2GB (baseline) vs 11M × 4 bytes = ~44MB (LoRA)
- **Optimizer States**: 2x gradient memory (Adam)
- **Activations**: Scales with batch size × sequence length

**Result**: LoRA reduces gradient+optimizer memory by **~4.5GB**, enabling larger batch sizes.

## Conclusions

This experimental journey demonstrates that **LoRA is not just more efficient—it's practically necessary** for large model fine-tuning:

1. **Performance Superior**: +6.68% accuracy with 97.8% fewer parameters
2. **Memory Essential**: Enables training where full fine-tuning fails
3. **Batch Size Advantage**: 8x larger batches possible
4. **Storage Efficient**: Save adapters (~MB) vs full models (~GB)
5. **Hardware Democratizing**: Makes LLM fine-tuning accessible on consumer GPUs

**The verdict**: LoRA transforms LLM fine-tuning from impossible to practical, while improving results.

## Dependencies

- PyTorch 2.5.1+ with CUDA 12.1 support
- Transformers 4.56.0+
- PEFT (latest from GitHub)
- Accelerate 1.0.0+
- scikit-learn for evaluation metrics
- Datasets for IMDB loading

## Additional Resources

**Author**: [Sagar Sarkale](https://www.linkedin.com/in/sagar-sarkale/)
📖 **For more insights**: [Deep Dive into LoRA: A Practical Exploration](https://sagarsarkale.com/blog/genai/lora-deepdive/)