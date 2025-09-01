```log
Llama 3.2 1B Sentiment Analysis - Baseline
============================================================
Loading IMDB dataset...
Train: 4000, Val: 1000, Test: 2500
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
--------------------------------------------------

Comprehensive Evaluation:
============================================================
Validating: 100%|██████████| 157/157 [02:07<00:00,  1.23it/s]
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


```log
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
--------------------------------------------------

Comprehensive Evaluation:
============================================================
Validating: 100%|██████████| 157/157 [02:19<00:00,  1.12it/s]
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

## GPU Memory Usage Analysis

**Why Both Scripts Use Similar GPU Memory:**

1. **Same Base Model Loading**: Both approaches load the complete `meta-llama/Llama-3.2-1B` model (~1.2B parameters) into GPU memory
   - Base model memory footprint remains constant regardless of training approach
   - Forward pass requires full model to be in memory

2. **LoRA Memory Characteristics**:
   - **Base model**: 1,235,818,498 parameters (always in GPU memory)
   - **LoRA adapters**: Only ~11M additional parameters (minimal overhead)
   - **Memory savings**: Come from reduced gradient storage, not model size

3. **Memory Breakdown**:
   - **Baseline**: Full model + gradients for 505M trainable parameters (40.9%)
   - **LoRA**: Full model + small adapters + gradients for 11M parameters (0.9%)

4. **Training Time Similarity**:
   - Baseline: 492.7s per epoch
   - LoRA: 496.6s per epoch
   - Similar because forward pass dominates compute time (same for both)

**LoRA's Real Benefits**:
- **Parameter Efficiency**: 97.8% reduction in trainable parameters (505M → 11M)
- **Storage Efficiency**: Save only LoRA weights (~MB) vs full model (~GB)
- **Better Performance**: 87.16% → 93.84% accuracy with fewer parameters
- **Reduced Overfitting**: Constrained parameter space leads to better generalization
- **Fine-tuning Speed**: Faster gradient updates during backpropagation

**Key Insight**: LoRA's memory efficiency comes from gradient computation, not model storage. The base transformer still needs full GPU memory for inference.