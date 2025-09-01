# %% [markdown]
# # Sentiment Analysis: Baseline Llama 3.2 1B
# ## Fine-tuning for Classification

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
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

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv(".env")
# HF_TOKEN=hf_xxxxxx
# make sure to set the HF_TOKEN in the .env file


# %%
@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = 16
    max_length: int = 256
    num_classes: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 1
    early_stopping_patience: int = 2

    warmup_steps: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: int = 5000  # Limit dataset size for faster training


config = Config()


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
            "labels": torch.tensor(label, dtype=torch.long),
        }


# %%
class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._prepare_datasets()

    def _prepare_datasets(self):
        print("Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        # Limit dataset size for faster training
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

        # Split train into train/val
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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load the base model
        self.llama = AutoModel.from_pretrained(config.model_name)

        # Resize token embeddings if needed
        if (
            len(AutoTokenizer.from_pretrained(config.model_name))
            != self.llama.config.vocab_size
        ):
            self.llama.resize_token_embeddings(
                len(AutoTokenizer.from_pretrained(config.model_name))
            )

        # Replace the last layer with classification head
        hidden_size = self.llama.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_size, config.num_classes)
        )

        # Freeze some layers to reduce memory usage
        self._freeze_layers()

    def _freeze_layers(self):
        # Freeze first 12 layers (out of ~22), keep last layers trainable
        layers_to_freeze = 12
        for i, layer in enumerate(self.llama.layers):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state
        last_hidden_states = outputs.last_hidden_state

        # Use the last token's representation (like GPT)
        batch_size = last_hidden_states.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled_output = last_hidden_states[range(batch_size), sequence_lengths]

        # Classification
        logits = self.classifier(pooled_output)
        return logits


# %%
class Trainer:
    def __init__(self, model: nn.Module, config: Config, total_steps: int):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        # Only optimize unfrozen parameters
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

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training"):
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

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Validating"):
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

    def detailed_evaluation(self, dataloader: DataLoader, split_name: str = "Test"):
        """Comprehensive evaluation with detailed metrics"""
        loss, accuracy, predictions, labels = self.validate(dataloader)

        # Calculate detailed metrics
        f1 = f1_score(labels, predictions, average="weighted")
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")

        # Classification report
        class_report = classification_report(
            labels, predictions, target_names=["Negative", "Positive"], output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        results = {
            "split": split_name,
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

        return results

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training on {self.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
        )

        for epoch in range(self.config.max_epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch + 1}/{self.config.max_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ✓ New best model (loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
            print("-" * 50)


# %%
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("Llama 3.2 1B Sentiment Analysis - Baseline")
    print("=" * 60)

    # Data
    data_manager = DataManager(config)
    train_loader = data_manager.get_dataloader("train")
    val_loader = data_manager.get_dataloader("val")
    test_loader = data_manager.get_dataloader("test")

    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config.max_epochs

    # Model and training
    model = LlamaClassifier(config)
    trainer = Trainer(model, config, total_steps)

    # Train
    trainer.fit(train_loader, val_loader)

    # Comprehensive evaluation
    print("\nComprehensive Evaluation:")
    print("=" * 60)

    # Test evaluation
    test_results = trainer.detailed_evaluation(test_loader, "Test")

    print(f"Test Results:")
    print(f"  Loss: {test_results['loss']:.4f}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  F1 Score: {test_results['f1_score']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")

    print(f"\nModel Parameters:")
    print(f"  Total: {test_results['total_params']:,}")
    print(
        f"  Trainable: {test_results['trainable_params']:,} ({100 * test_results['trainable_params'] / test_results['total_params']:.1f}%)"
    )

    print(f"\nPer-Class Results:")
    for class_name, metrics in test_results["classification_report"].items():
        if class_name in ["Negative", "Positive"]:
            print(
                f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}"
            )

    # Save results
    results_file = f"results_baseline_{int(time.time())}.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            k: v for k, v in test_results.items() if k != "confusion_matrix"
        }
        serializable_results["confusion_matrix"] = test_results["confusion_matrix"]
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("=" * 60)

# %%
