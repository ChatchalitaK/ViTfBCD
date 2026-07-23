"""
Training loop for ViTfBCD
Handles: training, validation, early stopping, LR scheduling, checkpointing
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

try:
    from sklearn.metrics import f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[Warning] scikit-learn not installed. Macro-F1 tracking disabled, falling back to accuracy.")
    print("Install with: pip install scikit-learn")

from src.model import ViTfBCD


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with optional class weights + label smoothing.
    Down-weights easy/majority-class examples (e.g. ductal_carcinoma) so the model
    keeps learning from hard/minority-class examples instead of coasting on the
    dominant class. Works alongside (not instead of) the dataset's WeightedRandomSampler
    -- default alpha=None so the two mechanisms don't double-correct the imbalance.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# Matches dataset.py's SUBTYPE_TO_IDX ordering exactly:
# 0=adenosis, 1=fibroadenoma, 2=phyllodes_tumor, 3=tubular_adenoma (benign)
# 4=ductal_carcinoma, 5=lobular_carcinoma, 6=mucinous_carcinoma, 7=papillary_carcinoma (malignant)
BENIGN_CLASS_INDICES = (0, 1, 2, 3)
MALIGNANT_CLASS_INDICES = (4, 5, 6, 7)


class CostSensitiveLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, num_classes: int = 8, danger_weight: float = 2.0,
                 benign_indices=BENIGN_CLASS_INDICES, malignant_indices=MALIGNANT_CLASS_INDICES):
        super().__init__()
        self.base_criterion = base_criterion
        self.danger_weight = danger_weight
        self.malignant_indices = malignant_indices
        mask = torch.zeros(num_classes)
        for i in benign_indices:
            mask[i] = 1.0
        self.benign_mask = mask  

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_criterion(logits, targets)

        if self.benign_mask.device != logits.device:
            self.benign_mask = self.benign_mask.to(logits.device)

        probs = torch.softmax(logits, dim=1)
        benign_prob_mass = probs @ self.benign_mask.to(probs.dtype)  

        is_malignant = torch.zeros_like(targets, dtype=torch.bool)
        for c in self.malignant_indices:
            is_malignant |= (targets == c)

        if is_malignant.any():
            danger_penalty = (benign_prob_mass * is_malignant.float()).sum() / is_malignant.float().sum()
        else:
            danger_penalty = benign_prob_mass.new_tensor(0.0)

        return base_loss + self.danger_weight * danger_penalty


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Returns mixed inputs, pairs of targets, and lambda. Set alpha<=0 to disable.
    Helps both class imbalance (soft-labels smooth out majority-class overconfidence)
    and the val-loss-plateau/overfitting problem seen in earlier training runs.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


class EarlyStopping:
    """
    mode="min": stop when `value` (e.g. val_loss) stops decreasing.
    mode="max": stop when `value` (e.g. macro_f1, acc) stops increasing.
    """
    def __init__(self, patience: int = 10, delta: float = 1e-4, mode: str = "min"):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        score = -value if self.mode == "min" else value
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Full training pipeline for ViTfBCD, ResNet50Baseline, InceptionV3Baseline,
    EfficientNetB4Baseline, or IRv2Baseline (see cnn_models.py).

    Args:
        model       : ViTfBCD or CNN baseline instance
        config      : training hyperparameter dict
        device      : torch device
        output_dir  : directory to save checkpoints and logs
    """

    def __init__(
        self,
        model,
        config: dict,
        device: torch.device,
        output_dir: str = "/home/user/Proj-Ploy/vit_breast_cancer/outputs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aux_loss_weight = config.get("aux_loss_weight", 0.4)

        self.use_focal_loss = config.get("use_focal_loss", True)
        self.focal_gamma = config.get("focal_gamma", 2.0)
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.focal_alpha_mode = config.get("focal_alpha_mode", "none")  # "none" | "sqrt_inv_freq" | "inv_freq"

        self.use_cost_sensitive_loss = config.get("use_cost_sensitive_loss", False)
        self.danger_weight = config.get("danger_weight", 2.0)
        model_num_classes = getattr(model, "num_classes", None)
        if self.use_cost_sensitive_loss and model_num_classes != 8:
            print(f"[Warning] use_cost_sensitive_loss=True but model.num_classes="
                  f"{model_num_classes} (expected 8 for dataset.py's subtype "
                  f"ordering) -- disabling cost-sensitive loss.")
            self.use_cost_sensitive_loss = False
        self.model_num_classes = model_num_classes or 8

        self.criterion = self._build_criterion(alpha=None)

        # MixUp — set mixup_alpha=0 in config to disable.
        self.mixup_alpha = config.get("mixup_alpha", 0.4)

        self.checkpoint_metric = config.get("checkpoint_metric", "macro_f1")
        if self.checkpoint_metric == "macro_f1" and not _SKLEARN_AVAILABLE:
            self.checkpoint_metric = "acc"

        self.early_stop_metric = config.get("early_stop_metric", self.checkpoint_metric)
        es_mode = "max" if self.early_stop_metric in ("macro_f1", "acc") else "min"

        base_lr = config.get("lr", 1e-4)
        weight_decay = config.get("weight_decay", 1e-4)

        if hasattr(self.model, "get_layered_parameters"):
            param_groups = self.model.get_layered_parameters()
            
            optimizer_params = [
                {
                    "params": param_groups["backbone"], 
                    "lr": base_lr * 0.1, 
                    "weight_decay": weight_decay
                },
                {
                    "params": param_groups["head"], 
                    "lr": base_lr, 
                    "weight_decay": weight_decay * 0.2  
                }
            ]
            print(f"[Trainer Dynamic LRs] Active strategy:")
            print(f"   ├─ Backbone Modules -> Learning Rate: {base_lr * 0.1:.2e} | Weight Decay: {weight_decay}")
            print(f"   └─ Classification Head -> Learning Rate: {base_lr:.2e} | Weight Decay: {weight_decay * 0.2}")
        else:
           
            optimizer_params = [p for p in model.parameters() if p.requires_grad]
            print(f"[Trainer Standard LR] Fallback active -> Universal Learning Rate: {base_lr:.2e}")

        self.optimizer = AdamW(
            optimizer_params,
            lr=base_lr
        )

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        self.lr_schedule_type = config.get("lr_schedule", "cosine")

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        self.lr_schedule_type = config.get("lr_schedule", "cosine")
        if self.lr_schedule_type == "plateau":
            plateau_mode = "max" if es_mode == "max" else "min"
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=plateau_mode,
                factor=config.get("lr_factor", 0.5),
                patience=config.get("lr_patience", 3),
                min_lr=config.get("min_lr", 1e-6),
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("epochs", 30),
                eta_min=config.get("min_lr", 1e-6),
            )

        # Mixed precision scaler (GPU only)
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.use_amp = device.type == "cuda"

        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            delta=config.get("es_delta", 1e-4),
            mode=es_mode,
        )

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

    def _build_criterion(self, alpha: Optional[torch.Tensor]) -> nn.Module:
        if self.use_focal_loss:
            base = FocalLoss(alpha=alpha, gamma=self.focal_gamma, label_smoothing=self.label_smoothing)
        else:
            base = nn.CrossEntropyLoss(weight=alpha, label_smoothing=self.label_smoothing)
        if self.use_cost_sensitive_loss:
            return CostSensitiveLoss(base, num_classes=self.model_num_classes, danger_weight=self.danger_weight)
        return base

    def _compute_class_alpha(self, train_loader: DataLoader) -> Optional[torch.Tensor]:
        """
        Build a per-class weight tensor for the loss based on train_loader's
        class_counts, honoring self.focal_alpha_mode. Default "none" because the
        dataset's WeightedRandomSampler is already correcting the imbalance at
        the sampling level -- adding a second full correction here would
        over-correct. "sqrt_inv_freq" gives a gentle second nudge if the sampler
        alone isn't enough; "inv_freq" is the full correction (use sampler OR
        this, not both at full strength).
        """
        if self.focal_alpha_mode == "none":
            return None
        if not hasattr(train_loader.dataset, "class_counts"):
            print("[Warning] train_loader.dataset has no class_counts; skipping alpha weighting.")
            return None

        counts = train_loader.dataset.class_counts
        num_classes = self.model.num_classes if hasattr(self.model, "num_classes") else max(counts.keys()) + 1
        total = sum(counts.values())
        weights = torch.ones(num_classes, dtype=torch.float32)
        for cls in range(num_classes):
            n = counts.get(cls, 0)
            if n == 0:
                continue
            inv_freq = total / n
            weights[cls] = inv_freq ** 0.5 if self.focal_alpha_mode == "sqrt_inv_freq" else inv_freq
        # Normalize so mean weight ~= 1 (keeps overall loss scale stable)
        weights = weights / weights.mean()
        return weights.to(self.device)

    # ── Single epoch ──────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float, Optional[float]]:
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # MixUp only during training; val/test always evaluated on real labels.
                if training and self.mixup_alpha > 0:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                else:
                    labels_a, labels_b, lam = labels, labels, 1.0

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)

                    if isinstance(outputs, tuple):
                        logits, aux_logits = outputs
                        main_loss = (
                            lam * self.criterion(logits, labels_a)
                            + (1 - lam) * self.criterion(logits, labels_b)
                        )
                        aux_loss = (
                            lam * self.criterion(aux_logits, labels_a)
                            + (1 - lam) * self.criterion(aux_logits, labels_b)
                        )
                        loss = main_loss + self.aux_loss_weight * aux_loss
                    else:
                        logits = outputs
                        loss = (
                            lam * self.criterion(logits, labels_a)
                            + (1 - lam) * self.criterion(logits, labels_b)
                        )

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    # Gradient clipping to stabilize transformer training
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                dominant_labels = labels_a if lam >= 0.5 else labels_b
                correct += (preds == dominant_labels).sum().item()
                total += images.size(0)

                if not training:
                    all_preds.append(preds.detach().cpu())
                    all_labels.append(labels.detach().cpu())

        macro_f1 = None
        if not training and _SKLEARN_AVAILABLE and all_preds:
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return total_loss / total, correct / total, macro_f1

    # ── Full training loop ────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        epochs = self.config.get("epochs", 30)

        alpha = self._compute_class_alpha(train_loader)
        self.criterion = self._build_criterion(alpha=alpha)

        cost_sensitive_str = f"CostSensitive(danger_weight={self.danger_weight})" if self.use_cost_sensitive_loss else "off"
        print(f"\n{'='*60}")
        print(f"  Training ViTfBCD-{self.config.get('model_size','base').capitalize()}")
        print(f"  Mode: {self.config.get('mode','binary')} | Epochs: {epochs}")
        print(f"  Device: {self.device} | AMP: {self.use_amp}")
        print(f"  Loss: {'FocalLoss' if self.use_focal_loss else 'CrossEntropyLoss'} "
              f"(gamma={self.focal_gamma}, alpha_mode={self.focal_alpha_mode}) | "
              f"Cost-sensitive: {cost_sensitive_str} | "
              f"MixUp alpha: {self.mixup_alpha} | Checkpoint metric: {self.checkpoint_metric} | "
              f"Early-stop metric: {self.early_stop_metric}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc, _ = self._run_epoch(train_loader, training=True)
            val_loss, val_acc, val_f1 = self._run_epoch(val_loader, training=False)

            if self.lr_schedule_type == "plateau":
                if self.early_stopping.mode == "max":
                    plateau_value = val_f1 if val_f1 is not None else val_acc
                else:
                    plateau_value = val_loss
                self.scheduler.step(plateau_value)
            else:
                self.scheduler.step()
            elapsed = time.time() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)

            lr = self.optimizer.param_groups[0]["lr"]
            lr_backbone = self.optimizer.param_groups[0]["lr"]
            lr_head = self.optimizer.param_groups[1]["lr"] if len(self.optimizer.param_groups) > 1 else lr_backbone
            
            f1_str = f"{val_f1:.4f}" if val_f1 is not None else "n/a"
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {f1_str} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s"
            )

            monitor_value = val_f1 if (self.checkpoint_metric == "macro_f1" and val_f1 is not None) else val_acc
            best_so_far = self.best_val_f1 if self.checkpoint_metric == "macro_f1" else self.best_val_acc
            if monitor_value > best_so_far:
                if self.checkpoint_metric == "macro_f1":
                    self.best_val_f1 = monitor_value
                self.best_val_acc = max(self.best_val_acc, val_acc)
                self._save_checkpoint("best_model.pt")
                metric_name = "macro-F1" if self.checkpoint_metric == "macro_f1" else "acc"
                print(f"  ✓ New best val {metric_name}: {monitor_value:.4f} — checkpoint saved")
            if self.early_stop_metric == "macro_f1":
                es_value = val_f1 if val_f1 is not None else val_acc
            elif self.early_stop_metric == "acc":
                es_value = val_acc
            else:
                es_value = val_loss
            if self.early_stopping(es_value):
                print(f"\n  Early stopping triggered at epoch {epoch} "
                      f"(no improvement in {self.early_stop_metric}).")
                break

        # Save final history
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.4f} | Best val macro-F1: {self.best_val_f1:.4f}")
        print(f"History saved to {history_path}")

        return self.history

    def _save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_f1": self.best_val_f1,
            "config": self.config,
        }, path)

    def load_best(self):
        ckpt_path = self.output_dir / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"]

        pos_key = "vit.encoder.pos_embedding"
        needs_resize = (
            pos_key in state_dict
            and hasattr(self.model, "resize_position_embeddings")
            and hasattr(self.model, "vit")
            and state_dict[pos_key].shape != self.model.vit.encoder.pos_embedding.shape
        )
        if needs_resize:
            print("[Info] Checkpoint position-embedding shape differs from the "
                  "current model — resizing before loading.")
            self.model.resize_position_embeddings()

        self.model.load_state_dict(state_dict)
        f1 = ckpt.get("best_val_f1")
        f1_str = f", val_f1={f1:.4f}" if f1 else ""
        print(f"Loaded best model (val_acc={ckpt['best_val_acc']:.4f}{f1_str})")