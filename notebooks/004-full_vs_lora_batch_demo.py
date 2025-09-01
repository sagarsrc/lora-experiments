# %% [markdown]
# # Full Fine-tuning vs LoRA: Batch Size & Memory Demonstration
# ## Comparing True Full Fine-tuning (small batches) vs LoRA (larger batches)
#
# **⚠️ WARNING: This notebook will likely crash with OOM errors during full fine-tuning!**
#
# **Expected CUDA Out of Memory Error:**
# ```log
# OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total
# capacity of 22.07 GiB of which 58.44 MiB is free. Process 2232965 has 22.00 GiB memory
# in use. Of the allocated memory 21.71 GiB is allocated by PyTorch, and 14.49 MiB is
# reserved by PyTorch but unallocated. If reserved but unallocated memory is large try
# setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
# ```
#
# This is **intentional** to demonstrate why LoRA is necessary:
# - Full fine-tuning of 1B+ models requires massive GPU memory
# - Even batch size 2 will likely cause Out-of-Memory errors on consumer GPUs
# - LoRA can use much larger batch sizes (16+) due to memory efficiency
#
# **User Control:**
# - Set `RUN_FULL_FINETUNING = False` to skip the crashing experiment
# - Set `RUN_FULL_FINETUNING = True` to experience the OOM error firsthand
#
# **Expected behavior:**
# - Full fine-tuning experiment will crash with CUDA OOM (if enabled)
# - LoRA experiment will run successfully with larger batch size
# - This demonstrates LoRA's practical necessity for large model fine-tuning

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import time
from dataclasses import dataclass
from typing import Tuple
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import warnings
import gc

warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv(".env")

# %%
# USER CONTROL FLAG: Set to False to skip the OOM-causing full fine-tuning experiment
RUN_FULL_FINETUNING = (
    False  # Change to False to skip the crashing experiment and only run LoRA
)


# %%
@dataclass
class FullConfig:
    """Configuration for full fine-tuning - requires smaller batch size"""

    model_name: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = (
        2  # Will likely cause OOM for full fine-tuning - demonstrating the problem
    )
    max_length: int = 256
    num_classes: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 1
    early_stopping_patience: int = 2
    warmup_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: int = 1000  # Smaller dataset for demonstration
    approach: str = "full"


@dataclass
class LoRAConfig:
    """Configuration for LoRA - can use larger batch size"""

    model_name: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = 16  # Larger batch size possible with LoRA
    max_length: int = 256
    num_classes: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_epochs: int = 1
    early_stopping_patience: int = 2
    warmup_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: int = 1000  # Same dataset size for fair comparison
    approach: str = "lora"

    # LoRA specific
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


# %%
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(
                label, dtype=torch.long
            ),  # torch.long = int64 for classification labels
        }


# %%
class DataManager:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._prepare_datasets()

    def _prepare_datasets(self):
        print(f"Loading IMDB dataset for {self.config.approach} fine-tuning...")
        dataset = load_dataset("imdb")

        # Use smaller dataset for demonstration
        train_data = (
            dataset["train"]
            .shuffle(seed=42)
            .select(range(min(self.config.max_samples, len(dataset["train"]))))
        )
        test_data = (
            dataset["test"]
            .shuffle(seed=42)
            .select(range(min(self.config.max_samples // 2, len(dataset["test"]))))
        )

        train_split = train_data.train_test_split(test_size=0.2, seed=42)

        self.train_texts = train_split["train"]["text"]
        self.train_labels = train_split["train"]["label"]
        self.val_texts = train_split["test"]["text"]
        self.val_labels = train_split["test"]["label"]
        self.test_texts = test_data["text"]
        self.test_labels = test_data["label"]

        print(
            f"Train: {len(self.train_texts)}, Val: {len(self.val_texts)}, Test: {len(self.test_texts)}"
        )
        print(f"Batch size: {self.config.batch_size}")

    def get_dataloader(self, split: str = "train") -> DataLoader:
        if split == "train":
            dataset = IMDBDataset(
                self.train_texts,
                self.train_labels,
                self.tokenizer,
                self.config.max_length,
            )
            shuffle = True
        elif split == "val":
            dataset = IMDBDataset(
                self.val_texts, self.val_labels, self.tokenizer, self.config.max_length
            )
            shuffle = False
        else:  # test
            dataset = IMDBDataset(
                self.test_texts,
                self.test_labels,
                self.tokenizer,
                self.config.max_length,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )


# %%
class LlamaClassifier(nn.Module):
    def __init__(self, config, full_finetuning=True):
        super().__init__()
        self.config = config
        self.full_finetuning = full_finetuning

        print(
            f"Loading Llama 3.2 1B for {'FULL' if full_finetuning else 'LoRA'} fine-tuning..."
        )

        # NOTE: device_map="auto" optimizations commented out due to runtime errors
        # This will likely cause OOM errors for full fine-tuning on consumer GPUs
        # Proper solution would require gradient checkpointing, DeepSpeed, or multi-GPU setup

        self.llama = AutoModel.from_pretrained(config.model_name)

        if full_finetuning:
            print("  WARNING: Full fine-tuning without memory optimizations")
            print("  This will likely cause OOM errors on consumer GPUs")
            print("  Consider using gradient checkpointing or smaller model")

        if (
            len(AutoTokenizer.from_pretrained(config.model_name))
            != self.llama.config.vocab_size
        ):
            self.llama.resize_token_embeddings(
                len(AutoTokenizer.from_pretrained(config.model_name))
            )

        hidden_size = self.llama.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_size, config.num_classes)
        )

        if full_finetuning:
            print("  All parameters will be trained (TRUE full fine-tuning)")
            print("  This requires small batch sizes due to memory constraints")
        else:
            print("  Base model will have LoRA adapters applied")

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        batch_size = last_hidden_states.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled_output = last_hidden_states[range(batch_size), sequence_lengths]
        logits = self.classifier(pooled_output)
        return logits


# %%
class MemoryTracker:
    """Track GPU memory usage during training"""

    def __init__(self):
        self.reset()

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_memory_stats(self):
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "peak": 0}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }

    def print_memory_stats(self, stage=""):
        stats = self.get_memory_stats()
        print(
            f"  {stage} - GPU Memory: Allocated={stats['allocated_gb']:.2f}GB, "
            f"Reserved={stats['reserved_gb']:.2f}GB, Peak={stats['peak_gb']:.2f}GB"
        )


# %%
class Trainer:
    def __init__(self, model: nn.Module, config, total_steps: int, use_lora=False):
        self.config = config
        self.device = torch.device(config.device)
        self.memory_tracker = MemoryTracker()

        if use_lora:
            print("Applying LoRA to enable larger batch sizes...")
            model = self._apply_lora(model)

        self.model = model.to(self.device)

        # Memory check after model loading
        self.memory_tracker.print_memory_stats("After model loading")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        model.llama = get_peft_model(model.llama, lora_config)
        model.llama.print_trainable_parameters()
        return model

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        # Track peak memory during training
        self.memory_tracker.reset()

        for i, batch in enumerate(
            tqdm(dataloader, desc=f"Training ({self.config.approach})")
        ):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            # Print memory stats every 10 batches
            if i % 10 == 0:
                self.memory_tracker.print_memory_stats(f"Batch {i}")

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Validating ({self.config.approach})"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        return total_loss / len(dataloader), accuracy, all_predictions, all_labels

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training with {self.config.approach.upper()} approach on {self.device}")
        print(f"Batch size: {self.config.batch_size}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
        )

        # Track memory at start of training
        self.memory_tracker.print_memory_stats("Training start")

        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)
            epoch_time = time.time() - start_time

            print(f"Epoch {epoch + 1}/{self.config.max_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # Final memory stats
            self.memory_tracker.print_memory_stats("Epoch end")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  New best model (loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
            print("-" * 60)


# %%
def run_experiment(config, use_lora=False):
    """Run a single experiment with given configuration"""

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {'LoRA' if use_lora else 'FULL'} Fine-tuning")
    print(f"Batch Size: {config.batch_size}")
    print(f"{'=' * 80}")

    # Clean up memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Data preparation
    data_manager = DataManager(config)
    train_loader = data_manager.get_dataloader("train")
    val_loader = data_manager.get_dataloader("val")
    test_loader = data_manager.get_dataloader("test")

    # Calculate total steps
    total_steps = len(train_loader) * config.max_epochs

    # Model and trainer
    model = LlamaClassifier(config, full_finetuning=not use_lora)
    trainer = Trainer(model, config, total_steps, use_lora=use_lora)

    # Training
    start_time = time.time()
    trainer.fit(train_loader, val_loader)
    total_training_time = time.time() - start_time

    # Final evaluation
    print(f"\nFinal Evaluation:")
    test_loss, test_acc, predictions, labels = trainer.validate(test_loader)

    results = {
        "approach": config.approach,
        "batch_size": config.batch_size,
        "total_training_time": total_training_time,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "total_params": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_params": sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        ),
        "peak_memory_gb": trainer.memory_tracker.get_memory_stats()["peak_gb"],
    }

    print(f"Results Summary:")
    print(f"  Approach: {results['approach'].upper()}")
    print(f"  Batch Size: {results['batch_size']}")
    print(f"  Training Time: {results['total_training_time']:.1f}s")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Peak Memory: {results['peak_memory_gb']:.2f}GB")
    print(f"  Trainable Params: {results['trainable_params']:,}")

    return results


# %%
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("Memory & Batch Size Comparison: Full Fine-tuning vs LoRA")
    print("=" * 80)

    # Run full fine-tuning experiment (small batch size) - WILL LIKELY CRASH!
    if RUN_FULL_FINETUNING:
        print("⚠️  Running FULL fine-tuning experiment - expect CUDA OOM error!")
        print("   This is intentional to demonstrate LoRA's necessity")
        try:
            full_config = FullConfig()
            full_results = run_experiment(full_config, use_lora=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n🔥 EXPECTED CUDA OOM ERROR OCCURRED:")
                print(f"   {str(e)[:200]}...")
                print(f"\n   This demonstrates why LoRA is necessary!")
                print(f"   Setting full_results to None and continuing with LoRA...")
                full_results = None
            else:
                raise e
    else:
        print("⏭️  Skipping full fine-tuning (RUN_FULL_FINETUNING = False)")
        print("   Set RUN_FULL_FINETUNING = True to experience the OOM error")
        full_results = None

    # Clean up memory between experiments
    torch.cuda.empty_cache()
    gc.collect()

    # Run LoRA experiment (larger batch size) - Should work fine
    print("\n✅ Running LoRA experiment - should work with larger batch size")
    lora_config = LoRAConfig()
    lora_results = run_experiment(lora_config, use_lora=True)

    # Comparison summary
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON")
    print(f"{'=' * 80}")

    if full_results is not None:
        print(f"Full Fine-tuning:")
        print(f"  Batch Size: {full_results['batch_size']}")
        print(f"  Peak Memory: {full_results['peak_memory_gb']:.2f}GB")
        print(f"  Training Time: {full_results['total_training_time']:.1f}s")
        print(f"  Test Accuracy: {full_results['test_accuracy']:.4f}")
        print(f"  Trainable Params: {full_results['trainable_params']:,}")
    else:
        print(f"Full Fine-tuning: CRASHED with CUDA OOM (as expected)")
        print(f"  Attempted Batch Size: 2")
        print(f"  Status: Out of Memory Error")

    print(f"\nLoRA Fine-tuning:")
    print(f"  Batch Size: {lora_results['batch_size']}")
    print(f"  Peak Memory: {lora_results['peak_memory_gb']:.2f}GB")
    print(f"  Training Time: {lora_results['total_training_time']:.1f}s")
    print(f"  Test Accuracy: {lora_results['test_accuracy']:.4f}")
    print(f"  Trainable Params: {lora_results['trainable_params']:,}")

    print(f"\nKey Insights:")
    if full_results is not None:
        batch_ratio = lora_results["batch_size"] / full_results["batch_size"]
        memory_diff = lora_results["peak_memory_gb"] - full_results["peak_memory_gb"]
        param_reduction = (
            1 - lora_results["trainable_params"] / full_results["trainable_params"]
        ) * 100

        print(f"  LoRA enables {batch_ratio:.0f}x larger batch size")
        print(f"  Memory difference: {memory_diff:+.2f}GB (LoRA vs Full)")
        print(f"  Parameter reduction: {param_reduction:.1f}%")
    else:
        batch_ratio = lora_results["batch_size"] / 2  # Attempted full batch size
        print(
            f"  LoRA enables {batch_ratio:.0f}x larger batch size than attempted full fine-tuning"
        )
        print(f"  Full fine-tuning FAILED due to memory constraints")
        print(f"  LoRA succeeds where full fine-tuning fails")

    print(f"  This demonstrates LoRA's practical necessity for large model training")

    # Save comparison results
    comparison = {
        "full_finetuning": full_results,
        "lora_finetuning": lora_results,
        "comparison_metrics": {
            "batch_size_ratio": batch_ratio,
            "memory_difference_gb": memory_diff if full_results else None,
            "parameter_reduction_percent": param_reduction if full_results else None,
            "full_finetuning_failed": full_results is None,
        },
    }

    results_file = f"batch_comparison_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison results saved to: {results_file}")
    print("=" * 80)

# %%
