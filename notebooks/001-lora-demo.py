# %% [markdown]
# # MNIST Training: MLP vs MLP with LoRA
# ## Modular Implementation using PyTorch

# %%
# Installation and imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from functools import wraps
import time
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import os
from tqdm import tqdm
from datetime import datetime

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# %%
# Decorators for modularity
def timer(func):
    """Decorator to time function execution"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper


def log_shape(func):
    """Decorator to log tensor shapes during forward pass"""

    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        if hasattr(self, "debug") and self.debug:
            print(f"Input shape to {func.__name__}: {x.shape}")
        result = func(self, x, *args, **kwargs)
        if hasattr(self, "debug") and self.debug and isinstance(result, torch.Tensor):
            print(f"Output shape from {func.__name__}: {result.shape}")
        return result

    return wrapper


def validate_input(input_dim: int):
    """Decorator to validate input dimensions"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            # Skip validation during tracing (when debug is False or not set)
            if getattr(self, "debug", True) and not torch.jit.is_tracing():
                if x.shape[-1] != input_dim:
                    raise ValueError(
                        f"Expected input dim {input_dim}, got {x.shape[-1]}"
                    )
            return func(self, x, *args, **kwargs)

        return wrapper

    return decorator


def track_gradients(func):
    """Decorator to track gradient norms during backward pass"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if hasattr(self, "track_grads") and self.track_grads:
            grad_norms = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()
            self.gradient_history.append(grad_norms)
        return result

    return wrapper


def auto_device(func):
    """Decorator to automatically move tensors to the correct device"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        device = next(self.parameters()).device
        args = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        kwargs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return func(self, *args, **kwargs)

    return wrapper


# %%
# Configuration
@dataclass
class Config:
    # Data
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = "./data"

    # Model
    input_dim: int = 784  # 28x28 flattened
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    output_dim: int = 10
    dropout_rate: float = 0.2

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    early_stopping_patience: int = 5
    gradient_clip: float = 1.0

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["linear"])

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    save_dir: str = "../checkpoints"
    tensorboard_dir: str = "../runs"


config = Config()


# %%
# Data Module with decorators
class MNISTDataManager:
    def __init__(self, config: Config):
        self.config = config
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten for MLP
            ]
        )
        self._prepare_datasets()

    @timer
    def _prepare_datasets(self):
        """Download and prepare MNIST datasets"""
        # Training data
        mnist_full = datasets.MNIST(
            self.config.data_dir, train=True, download=True, transform=self.transform
        )

        # Split into train and validation
        train_size = int(0.9 * len(mnist_full))
        val_size = len(mnist_full) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            mnist_full, [train_size, val_size]
        )

        # Test data
        self.test_dataset = datasets.MNIST(
            self.config.data_dir, train=False, download=True, transform=self.transform
        )

    def get_dataloader(self, split: str = "train") -> DataLoader:
        """Get dataloader for specified split"""
        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

        shuffle = split == "train"
        dataset = dataset_map[split]

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )


# %%
# Base MLP Model with decorators
class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.debug = False

        # Add a model config dict for PEFT compatibility
        self.model_config = {
            "tie_word_embeddings": False,
            "vocab_size": config.output_dim,
            "hidden_size": config.hidden_dims[-1]
            if config.hidden_dims
            else config.input_dim,
        }

        # Build hidden layers
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(MLPBlock(prev_dim, hidden_dim, config.dropout_rate))
            prev_dim = hidden_dim

        self.blocks = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, config.output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @validate_input(784)
    @log_shape
    def forward(self, x):
        x = self.blocks(x)
        x = self.output(x)
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_stats(self) -> Dict[str, float]:
        """Get statistics about model parameters"""
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.extend(p.detach().cpu().numpy().flatten())

        return {
            "mean": np.mean(params),
            "std": np.std(params),
            "min": np.min(params),
            "max": np.max(params),
        }

    def get_model_config(self):
        """Return model config for PEFT compatibility"""
        return self.model_config


# %%
# Model Architecture Printer
class ModelPrinter:
    """Utility class to print model architecture and statistics"""

    @staticmethod
    def print_model_architecture(model: nn.Module, model_name: str = "Model"):
        """Print detailed model architecture"""
        print("\n" + "=" * 70)
        print(f"{model_name} Architecture")
        print("=" * 70)

        # Check if it's a PEFT model
        is_peft = hasattr(model, "peft_config")

        if is_peft:
            print("Type: LoRA-Enhanced Model")
            print("\nLoRA Configuration:")
            for key, value in model.peft_config.items():
                print(f"  {key}: {value}")
            print("\nBase Model Architecture:")
            base_model = model.get_base_model()
            print(base_model)
        else:
            print("Type: Standard Model")
            print("\nModel Architecture:")
            print(model)

        print("\n" + "-" * 70)
        ModelPrinter.print_parameter_summary(model, is_peft)
        print("=" * 70)

    @staticmethod
    def print_parameter_summary(model: nn.Module, is_peft: bool = False):
        """Print parameter summary with layer-wise breakdown"""
        total_params = 0
        trainable_params = 0
        layer_info = []

        print("\nLayer-wise Parameter Count:")
        print("-" * 50)
        print(f"{'Layer Name':<40} {'Parameters':>10} {'Trainable':>10}")
        print("-" * 50)

        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count
                trainable = "Yes"
            else:
                trainable = "No"

            # Only show main layers, not individual weights/biases
            if any(x in name for x in ["weight", "bias"]):
                layer_name = name.rsplit(".", 1)[0] if "." in name else name
                param_type = name.rsplit(".", 1)[1] if "." in name else ""

                # Aggregate by layer
                found = False
                for info in layer_info:
                    if info["name"] == layer_name:
                        info["params"] += param_count
                        if param.requires_grad:
                            info["trainable"] += param_count
                        found = True
                        break

                if not found:
                    layer_info.append(
                        {
                            "name": layer_name,
                            "params": param_count,
                            "trainable": param_count if param.requires_grad else 0,
                        }
                    )

        # Print aggregated layer info
        for info in layer_info:
            trainable_str = "Yes" if info["trainable"] > 0 else "No"
            print(f"{info['name']:<40} {info['params']:>10,} {trainable_str:>10}")

        print("-" * 50)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")

        if is_peft:
            efficiency = (trainable_params / total_params) * 100
            print(f"Training Efficiency: {efficiency:.2f}% of parameters are trainable")

        # Memory estimation
        param_size = total_params * 4 / (1024**2)  # Assuming float32
        print(f"\nEstimated Model Size: {param_size:.2f} MB (float32)")


# %%
# Auto Trainer with decorators
class AutoTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        use_lora: bool = False,
        experiment_name: str = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.use_lora = use_lora

        # Apply LoRA if needed
        if use_lora:
            model = self._apply_lora(model)

        self.model = model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.max_epochs)

        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_val_acc = 0
        self.best_val_loss = float(
            "inf"
        )  # Initialize with infinity for loss-based early stopping
        self.patience_counter = 0
        self.global_step = 0

        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.tensorboard_dir, exist_ok=True)

        # Setup TensorBoard
        if experiment_name is None:
            experiment_name = f"{'lora' if use_lora else 'baseline'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.writer = SummaryWriter(
            os.path.join(config.tensorboard_dir, experiment_name)
        )

        # Log model graph
        self._log_model_graph()

    def _log_model_graph(self):
        """Log model graph to TensorBoard"""
        try:
            # Temporarily disable debug mode to avoid shape logging during tracing
            original_debug = getattr(self.model, "debug", False)
            if hasattr(self.model, "debug"):
                self.model.debug = False

            dummy_input = torch.randn(1, self.config.input_dim).to(self.device)
            self.writer.add_graph(self.model, dummy_input)

            # Restore debug mode
            if hasattr(self.model, "debug"):
                self.model.debug = original_debug

        except Exception as e:
            print(f"Could not log model graph: {e}")

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to the model"""
        # Debug: Print available modules
        print("Available modules for LoRA:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  - {name}: {type(module)}")

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    @timer
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{self.config.max_epochs} [Train]"
        )

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            self.optimizer.step()

            # Metrics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            pbar.set_postfix(
                {"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
            )

            # Log to TensorBoard
            if batch_idx % self.config.log_interval == 0:
                self.writer.add_scalar("Train/Loss_step", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "Train/Acc_step", 100.0 * correct / total, self.global_step
                )
                self.writer.add_scalar(
                    "Train/LR", self.optimizer.param_groups[0]["lr"], self.global_step
                )

                # Log gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        self.writer.add_histogram(
                            f"Gradients/{name}", param.grad, self.global_step
                        )

            self.global_step += 1

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, split: str = "val") -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"[{split.capitalize()}]")

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            pbar.set_postfix(
                {"loss": running_loss / (len(pbar) + 1), "acc": 100.0 * correct / total}
            )

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    @timer
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """Full training loop with early stopping"""
        print(f"\nStarting training on {self.device}")
        print(f"Model: {'LoRA-enhanced' if self.use_lora else 'Standard'} MLP")
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )
        print("=" * 50)

        for epoch in range(self.config.max_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.metrics["train_loss"].append(train_metrics["loss"])
            self.metrics["train_acc"].append(train_metrics["acc"])

            # Validate
            val_metrics = self.validate(val_loader, "val")
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["val_acc"].append(val_metrics["acc"])

            # Learning rate scheduling
            self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalar("Train/Loss_epoch", train_metrics["loss"], epoch)
            self.writer.add_scalar("Train/Acc_epoch", train_metrics["acc"], epoch)
            self.writer.add_scalar("Val/Loss_epoch", val_metrics["loss"], epoch)
            self.writer.add_scalar("Val/Acc_epoch", val_metrics["acc"], epoch)

            # Log weight distributions
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"Weights/{name}", param, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}"
            )
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}"
            )
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping and checkpointing (use loss for better model selection)
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc = val_metrics["acc"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["acc"])
                print(
                    f"New best model saved - Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
                )
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break

            print("-" * 50)

        # Close TensorBoard writer
        self.writer.close()
        print(f"\nTensorBoard logs saved to: {self.writer.log_dir}")
        print(f"Run 'tensorboard --logdir {self.config.tensorboard_dir}' to view")

        return self.metrics

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config,
        }

        filename = f"{'lora' if self.use_lora else 'baseline'}_epoch{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}.pt"
        path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["val_acc"]


# %%
# Comparison and Visualization Utilities
class ModelComparator:
    def __init__(self, baseline_trainer: AutoTrainer, lora_trainer: AutoTrainer):
        self.baseline = baseline_trainer
        self.lora = lora_trainer

    def compare_parameters(self):
        """Compare parameter counts and final performance between models"""
        baseline_total = sum(p.numel() for p in self.baseline.model.parameters())
        baseline_trainable = sum(
            p.numel() for p in self.baseline.model.parameters() if p.requires_grad
        )

        lora_total = sum(p.numel() for p in self.lora.model.parameters())
        lora_trainable = sum(
            p.numel() for p in self.lora.model.parameters() if p.requires_grad
        )

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"Baseline MLP:")
        print(f"  Total parameters: {baseline_total:,}")
        print(f"  Trainable parameters: {baseline_trainable:,}")
        print(f"  Final Val Loss: {min(self.baseline.metrics['val_loss']):.4f}")
        print(f"  Final Val Accuracy: {max(self.baseline.metrics['val_acc']):.4f}")

        print(f"\nLoRA MLP:")
        print(f"  Total parameters: {lora_total:,}")
        print(f"  Trainable parameters: {lora_trainable:,}")
        print(f"  Final Val Loss: {min(self.lora.metrics['val_loss']):.4f}")
        print(f"  Final Val Accuracy: {max(self.lora.metrics['val_acc']):.4f}")

        print(f"\nParameter Efficiency:")
        param_reduction = (1 - lora_trainable / baseline_trainable) * 100
        print(f"  Trainable parameter reduction: {param_reduction:.2f}%")

        # Performance efficiency
        baseline_best_loss = min(self.baseline.metrics["val_loss"])
        lora_best_loss = min(self.lora.metrics["val_loss"])
        loss_improvement = (
            (baseline_best_loss - lora_best_loss) / baseline_best_loss
        ) * 100
        print(f"  Loss improvement: {loss_improvement:.2f}%")
        print(
            f"  Efficiency ratio: {param_reduction / abs(loss_improvement):.2f} (param reduction per % loss improvement)"
        )
        print("=" * 60)

    def plot_training_curves_seaborn(self):
        """Plot training curves comparison using Seaborn"""
        # Prepare data for seaborn
        epochs = list(range(len(self.baseline.metrics["train_loss"])))

        # Create DataFrame for plotting
        data_list = []
        for epoch in epochs:
            data_list.append(
                {
                    "Epoch": epoch,
                    "Loss": self.baseline.metrics["train_loss"][epoch],
                    "Accuracy": self.baseline.metrics["train_acc"][epoch],
                    "Type": "Train",
                    "Model": "Baseline",
                }
            )
            data_list.append(
                {
                    "Epoch": epoch,
                    "Loss": self.lora.metrics["train_loss"][epoch],
                    "Accuracy": self.lora.metrics["train_acc"][epoch],
                    "Type": "Train",
                    "Model": "LoRA",
                }
            )
            data_list.append(
                {
                    "Epoch": epoch,
                    "Loss": self.baseline.metrics["val_loss"][epoch],
                    "Accuracy": self.baseline.metrics["val_acc"][epoch],
                    "Type": "Validation",
                    "Model": "Baseline",
                }
            )
            data_list.append(
                {
                    "Epoch": epoch,
                    "Loss": self.lora.metrics["val_loss"][epoch],
                    "Accuracy": self.lora.metrics["val_acc"][epoch],
                    "Type": "Validation",
                    "Model": "LoRA",
                }
            )

        df = pd.DataFrame(data_list)

        # Create figure with seaborn
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Loss plots
        train_df = df[df["Type"] == "Train"]
        val_df = df[df["Type"] == "Validation"]

        # Training Loss
        sns.lineplot(
            data=train_df,
            x="Epoch",
            y="Loss",
            hue="Model",
            marker="o",
            ax=axes[0, 0],
            palette=["#1f77b4", "#ff7f0e"],
        )
        axes[0, 0].set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")

        # Training Accuracy
        sns.lineplot(
            data=train_df,
            x="Epoch",
            y="Accuracy",
            hue="Model",
            marker="s",
            ax=axes[0, 1],
            palette=["#1f77b4", "#ff7f0e"],
        )
        axes[0, 1].set_title(
            "Training Accuracy Comparison", fontsize=14, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")

        # Validation Loss
        sns.lineplot(
            data=val_df,
            x="Epoch",
            y="Loss",
            hue="Model",
            marker="o",
            ax=axes[1, 0],
            palette=["#2ca02c", "#d62728"],
        )
        axes[1, 0].set_title(
            "Validation Loss Comparison", fontsize=14, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")

        # Validation Accuracy
        sns.lineplot(
            data=val_df,
            x="Epoch",
            y="Accuracy",
            hue="Model",
            marker="s",
            ax=axes[1, 1],
            palette=["#2ca02c", "#d62728"],
        )
        axes[1, 1].set_title(
            "Validation Accuracy Comparison", fontsize=14, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")

        plt.suptitle(
            "Training Curves: Baseline vs LoRA", fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.show()

        # Additional statistical plot
        self.plot_performance_distribution()

        # Plot loss convergence analysis
        self.plot_loss_convergence()

    @timer
    def compare_inference_speed(self, dataloader: DataLoader, num_batches: int = 50):
        """Compare inference speeds"""

        def measure_speed(model, dataloader, num_batches):
            model.eval()
            times = []

            with torch.no_grad():
                for i, (data, _) in enumerate(dataloader):
                    if i >= num_batches:
                        break

                    data = data.to(next(model.parameters()).device)
                    start = time.time()
                    _ = model(data)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append(time.time() - start)

            return np.mean(times[5:])  # Skip warmup

        baseline_time = measure_speed(self.baseline.model, dataloader, num_batches)
        lora_time = measure_speed(self.lora.model, dataloader, num_batches)

        print("\n" + "=" * 60)
        print("INFERENCE SPEED COMPARISON")
        print("=" * 60)
        print(f"Baseline MLP: {baseline_time * 1000:.2f} ms/batch")
        print(f"LoRA MLP: {lora_time * 1000:.2f} ms/batch")
        print(f"Speedup: {baseline_time / lora_time:.2f}x")
        print("=" * 60)

    def plot_performance_distribution(self):
        """Plot performance distribution using violin plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Prepare data
        performance_data = pd.DataFrame(
            {
                "Baseline Train": self.baseline.metrics["train_acc"],
                "LoRA Train": self.lora.metrics["train_acc"],
                "Baseline Val": self.baseline.metrics["val_acc"],
                "LoRA Val": self.lora.metrics["val_acc"],
            }
        )

        # Melt for seaborn
        melted = performance_data.melt(var_name="Model_Type", value_name="Accuracy")
        melted["Model"] = melted["Model_Type"].apply(
            lambda x: "Baseline" if "Baseline" in x else "LoRA"
        )
        melted["Dataset"] = melted["Model_Type"].apply(
            lambda x: "Train" if "Train" in x else "Validation"
        )

        # Violin plot
        sns.violinplot(
            data=melted,
            x="Model",
            y="Accuracy",
            hue="Dataset",
            split=True,
            ax=axes[0],
            palette="muted",
        )
        axes[0].set_title(
            "Accuracy Distribution Comparison", fontsize=14, fontweight="bold"
        )
        axes[0].set_ylabel("Accuracy")

        # Box plot for loss comparison
        loss_data = pd.DataFrame(
            {
                "Baseline Train": self.baseline.metrics["train_loss"],
                "LoRA Train": self.lora.metrics["train_loss"],
                "Baseline Val": self.baseline.metrics["val_loss"],
                "LoRA Val": self.lora.metrics["val_loss"],
            }
        )

        melted_loss = loss_data.melt(var_name="Model_Type", value_name="Loss")
        melted_loss["Model"] = melted_loss["Model_Type"].apply(
            lambda x: "Baseline" if "Baseline" in x else "LoRA"
        )
        melted_loss["Dataset"] = melted_loss["Model_Type"].apply(
            lambda x: "Train" if "Train" in x else "Validation"
        )

        sns.boxplot(
            data=melted_loss,
            x="Model",
            y="Loss",
            hue="Dataset",
            ax=axes[1],
            palette="Set2",
        )
        axes[1].set_title(
            "Loss Distribution Comparison", fontsize=14, fontweight="bold"
        )
        axes[1].set_ylabel("Loss")

        plt.suptitle(
            "Performance Distribution Analysis", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def plot_loss_convergence(self):
        """Plot detailed loss convergence analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = list(range(len(self.baseline.metrics["train_loss"])))

        # Training loss convergence
        axes[0, 0].semilogy(
            epochs,
            self.baseline.metrics["train_loss"],
            label="Baseline",
            marker="o",
            alpha=0.7,
        )
        axes[0, 0].semilogy(
            epochs, self.lora.metrics["train_loss"], label="LoRA", marker="s", alpha=0.7
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Log Training Loss")
        axes[0, 0].set_title("Training Loss Convergence (Log Scale)", fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Validation loss convergence
        axes[0, 1].semilogy(
            epochs,
            self.baseline.metrics["val_loss"],
            label="Baseline",
            marker="o",
            alpha=0.7,
        )
        axes[0, 1].semilogy(
            epochs, self.lora.metrics["val_loss"], label="LoRA", marker="s", alpha=0.7
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Log Validation Loss")
        axes[0, 1].set_title(
            "Validation Loss Convergence (Log Scale)", fontweight="bold"
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Loss difference over time
        train_loss_diff = [
            baseline_loss - lora_loss
            for baseline_loss, lora_loss in zip(
                self.baseline.metrics["train_loss"], self.lora.metrics["train_loss"]
            )
        ]
        val_loss_diff = [
            baseline_loss - lora_loss
            for baseline_loss, lora_loss in zip(
                self.baseline.metrics["val_loss"], self.lora.metrics["val_loss"]
            )
        ]

        axes[1, 0].plot(epochs, train_loss_diff, marker="o", color="green", alpha=0.7)
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss Difference (Baseline - LoRA)")
        axes[1, 0].set_title("Training Loss Advantage", fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, val_loss_diff, marker="s", color="orange", alpha=0.7)
        axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss Difference (Baseline - LoRA)")
        axes[1, 1].set_title("Validation Loss Advantage", fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Cross-Entropy Loss Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

        # Print convergence statistics
        print("\n" + "=" * 50)
        print("LOSS CONVERGENCE ANALYSIS")
        print("=" * 50)
        final_train_diff = train_loss_diff[-1]
        final_val_diff = val_loss_diff[-1]
        print(f"Final Training Loss Advantage (LoRA): {-final_train_diff:.4f}")
        print(f"Final Validation Loss Advantage (LoRA): {-final_val_diff:.4f}")
        print(
            f"Average Loss Advantage (LoRA): {np.mean([-td for td in train_loss_diff]):.4f}"
        )
        print(
            f"Loss Convergence Rate (LoRA faster): {np.sum([1 for td in train_loss_diff if td > 0]) / len(train_loss_diff) * 100:.1f}% of epochs"
        )
        print("=" * 50)


# %%
# Visualization utilities
@timer
def visualize_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
):
    """Visualize model predictions on test samples"""
    model.eval()

    # Get a batch of data
    data, targets = next(iter(dataloader))
    data = data[:num_samples].to(device)
    targets = targets[:num_samples]

    # Get predictions
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.argmax(dim=1).cpu()

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = data[i].cpu().reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        color = "green" if predictions[i] == targets[i] else "red"
        axes[i].set_title(f"True: {targets[i]}, Pred: {predictions[i]}", color=color)
        axes[i].axis("off")

    plt.suptitle("Model Predictions on Test Samples", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_weight_distributions_seaborn(model: nn.Module, title: str):
    """Plot weight distribution using seaborn"""
    weights = []
    biases = []
    layer_weights = {}

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            layer_name = name.rsplit(".", 1)[0]
            layer_weights[layer_name] = param.detach().cpu().numpy().flatten()
            weights.extend(param.detach().cpu().numpy().flatten())
        elif "bias" in name and param.requires_grad:
            biases.extend(param.detach().cpu().numpy().flatten())

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Overall weight distribution
    ax1 = fig.add_subplot(gs[0, :])
    sns.histplot(
        weights,
        bins=50,
        kde=True,
        ax=ax1,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Density")
    ax1.set_title(
        f"Overall Weight Distribution - {title}", fontsize=14, fontweight="bold"
    )
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Zero")
    ax1.axvline(
        x=np.mean(weights),
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {np.mean(weights):.4f}",
    )
    ax1.legend()

    # Layer-wise weight distributions
    ax2 = fig.add_subplot(gs[1, 0])
    layer_data = []
    for layer_name, layer_w in layer_weights.items():
        for w in layer_w[:1000]:  # Sample for visualization
            layer_data.append(
                {
                    "Layer": layer_name.split(".")[-1]
                    if "." in layer_name
                    else layer_name,
                    "Weight": w,
                }
            )

    if layer_data:
        df_layers = pd.DataFrame(layer_data)
        sns.violinplot(data=df_layers, x="Layer", y="Weight", ax=ax2, palette="viridis")
        ax2.set_title("Layer-wise Weight Distributions", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Weight Value")
        ax2.tick_params(axis="x", rotation=45)

    # Bias distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if biases:
        sns.histplot(
            biases,
            bins=30,
            kde=True,
            ax=ax3,
            color="forestgreen",
            edgecolor="black",
            alpha=0.7,
        )
        ax3.set_xlabel("Bias Value")
        ax3.set_ylabel("Density")
        ax3.set_title(f"Bias Distribution - {title}", fontsize=12, fontweight="bold")
        ax3.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    # Statistical summary
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    stats_text = f"""
    {title} Weight Statistics:
    ────────────────────────────
    Mean: {np.mean(weights):.6f}    |    Std: {np.std(weights):.6f}
    Min:  {np.min(weights):.6f}    |    Max: {np.max(weights):.6f}
    25%:  {np.percentile(weights, 25):.6f}    |    75%: {np.percentile(weights, 75):.6f}
    Median: {np.median(weights):.6f}    |    Sparsity: {(np.abs(weights) < 0.01).mean() * 100:.2f}%
    """

    ax4.text(
        0.5,
        0.5,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        family="monospace",
    )

    plt.suptitle(f"Weight Analysis: {title}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# %%
# Main Execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("MNIST TRAINING: BASELINE MLP vs LoRA-ENHANCED MLP")
    print("=" * 60)

    # %%
    # Initialize data manager
    print("\nInitializing data...")
    data_manager = MNISTDataManager(config)

    # Get dataloaders
    train_loader = data_manager.get_dataloader("train")
    val_loader = data_manager.get_dataloader("val")
    test_loader = data_manager.get_dataloader("test")

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # %%
    # Create and train baseline model
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MLP")
    print("=" * 60)

    baseline_model = MLP(config)
    baseline_trainer = AutoTrainer(
        baseline_model, config, use_lora=False, experiment_name="baseline_mlp"
    )

    # Print baseline model architecture
    ModelPrinter.print_model_architecture(baseline_trainer.model, "Baseline MLP")

    baseline_metrics = baseline_trainer.fit(train_loader, val_loader)

    # %%
    # Create and train LoRA model
    print("\n" + "=" * 60)
    print("TRAINING LoRA-ENHANCED MLP")
    print("=" * 60)

    lora_model = MLP(config)
    lora_trainer = AutoTrainer(
        lora_model, config, use_lora=True, experiment_name="lora_mlp"
    )

    # Print LoRA model architecture
    ModelPrinter.print_model_architecture(lora_trainer.model, "LoRA-Enhanced MLP")

    lora_metrics = lora_trainer.fit(train_loader, val_loader)

    # %%
    # Test both models
    print("\n" + "=" * 60)
    print("TESTING PHASE")
    print("=" * 60)

    baseline_test_metrics = baseline_trainer.validate(test_loader, "test")
    lora_test_metrics = lora_trainer.validate(test_loader, "test")

    print(f"\nBaseline Test Accuracy: {baseline_test_metrics['acc']:.4f}")
    print(f"LoRA Test Accuracy: {lora_test_metrics['acc']:.4f}")

    # %%
    # Compare models
    comparator = ModelComparator(baseline_trainer, lora_trainer)
    comparator.compare_parameters()
    comparator.plot_training_curves_seaborn()
    comparator.compare_inference_speed(test_loader)

    # %%
    # Visualize predictions
    print("\nVisualizing Baseline Model Predictions...")
    visualize_predictions(baseline_trainer.model, test_loader, baseline_trainer.device)

    print("\nVisualizing LoRA Model Predictions...")
    visualize_predictions(lora_trainer.model, test_loader, lora_trainer.device)

    # %%
    # Plot weight distributions
    plot_weight_distributions_seaborn(baseline_trainer.model, "Baseline MLP")
    plot_weight_distributions_seaborn(lora_trainer.model, "LoRA MLP")

    # %%
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nDataset: MNIST (28x28 grayscale images, 10 classes)")
    print(f"Architecture: MLP with hidden dimensions {config.hidden_dims}")

    print("\nTraining Configuration:")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Max Epochs: {config.max_epochs}")
    print(f"  - Weight Decay: {config.weight_decay}")
    print(f"  - Gradient Clipping: {config.gradient_clip}")

    print("\nLoRA Configuration:")
    print(f"  - Rank: {config.lora_rank}")
    print(f"  - Alpha: {config.lora_alpha}")
    print(f"  - Dropout: {config.lora_dropout}")
    print(f"  - Target Modules: {config.lora_target_modules}")

    print("\nFinal Results:")
    print(f"  - Baseline Test Loss: {baseline_test_metrics['loss']:.4f}")
    print(f"  - Baseline Test Accuracy: {baseline_test_metrics['acc']:.4f}")
    print(f"  - LoRA Test Loss: {lora_test_metrics['loss']:.4f}")
    print(f"  - LoRA Test Accuracy: {lora_test_metrics['acc']:.4f}")
    print(f"  - Best Baseline Val Loss: {baseline_trainer.best_val_loss:.4f}")
    print(f"  - Best LoRA Val Loss: {lora_trainer.best_val_loss:.4f}")

    # Cross-entropy based efficiency metrics
    test_loss_improvement = (
        (baseline_test_metrics["loss"] - lora_test_metrics["loss"])
        / baseline_test_metrics["loss"]
    ) * 100
    print(f"  - Test Loss Improvement: {test_loss_improvement:.2f}%")

    baseline_params = sum(
        p.numel() for p in baseline_trainer.model.parameters() if p.requires_grad
    )
    lora_params = sum(
        p.numel() for p in lora_trainer.model.parameters() if p.requires_grad
    )
    print(
        f"\n  - Parameter Reduction: {(1 - lora_params / baseline_params) * 100:.2f}%"
    )
    print("=" * 70)
