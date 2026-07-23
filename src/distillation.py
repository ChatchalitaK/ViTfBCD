import time
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision.models as tvm

from src.dataset import _parse_patient_id  # SAME parser as everywhere else -- do not reimplement


#------------------------------------------------------------
# Student model factory
#------------------------------------------------------------
def build_student(student_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = student_name.lower().strip()

    if name == "mobilenet_v3_small":
        weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tvm.mobilenet_v3_small(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)

    elif name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)

        # Static Quantization
        model.quant = torch.ao.quantization.QuantStub()
        model.dequant = torch.ao.quantization.DeQuantStub()

        old_forward = model.forward
        def quantized_forward(x):
            x = model.quant(x)
            x = old_forward(x)
            return model.dequant(x)
        model.forward = quantized_forward

    elif name in ["deit_tiny", "deit_small"]:
        try:
            import timm
        except ImportError:
            raise ImportError("timm required for DeiT models. Install: pip install timm")
        
        timm_model_name = "deit_tiny_patch16_224" if name == "deit_tiny" else "deit_small_patch16_224"
        
        model = timm.create_model(
            timm_model_name, 
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=384,
            drop_path_rate=0.2,
        )
        print(f"  -> pos_embed shape after interpolation: {tuple(model.pos_embed.shape)}")

    else:
        raise ValueError(
            f"Unknown student '{student_name}'."
            "Choose: mobilenet_v3_small | efficientnet_b0 | deit_tiny | deit_small"
        )
    return model


#______________________________________________________________
# Distillation Loss
#______________________________________________________________
class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.00,
        alpha: float = 0.7,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.T         = temperature
        self.alpha     = alpha
        self.ce        = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels:         torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = self.ce(student_logits, labels)

        s_soft = F.log_softmax(student_logits / self.T, dim=1)
        t_soft = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.T ** 2)

        total = (self.alpha * ce_loss) + ((1.0 - self.alpha) * kd_loss)
        return total, ce_loss, kd_loss


#_____________________________________________________________
# Class weight helper
#_____________________________________________________________
def compute_class_weights(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in loader:
        labels = batch[1]
        counts += torch.bincount(labels, minlength=num_classes)

    counts = counts.float()
    total = counts.sum()
    print(f"  Class counts (train): {counts.tolist()}")

    weights = total / (num_classes * counts.clamp(min=1))
    weights = weights / weights.mean()
    print(f"  Class weights:        {[round(w, 3) for w in weights.tolist()]}")
    return weights.to(device)


#_____________________________________________________________
# DistillationTrainer
#_____________________________________________________________
class DistillationTrainer:
    def __init__(
            self,
            teacher: nn.Module,
            student: nn.Module,
            config:  dict,
            device:  torch.device,
            output_dir: str = "/home/user/Proj-Ploy/vit_breast_cancer/outputs/distill",
            num_classes: Optional[int] = None,
    ):
        self.teacher  = teacher.to(device).eval()
        self.student  = student.to(device)
        self.config   = config
        self.device   = device
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.class_weights = None
        self.criterion = DistillationLoss(
            temperature=config.get("temperature", 4.0),
            alpha=config.get("alpha", 0.7),
            class_weights=None,
            label_smoothing=config.get("label_smoothing", 0.1),
        )
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.get("lr", 5e-5),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        self.lr_schedule_type = config.get("lr_schedule", "plateau")
        if self.lr_schedule_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("epochs", 30),
                eta_min=config.get("min_lr", 1e-6)
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=config.get("lr_factor", 0.5),
                patience=config.get("lr_patience", 3),
                min_lr=config.get("min_lr", 1e-6),
            )

        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.use_amp = device.type == "cuda"
        self.history: Dict = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "val_patient_acc": [], "val_patient_acc_smoothed": [],
        }
        self.best_val_patient_acc = 0.0
        # Checkpoint/LR-plateau decisions use a k-epoch moving average of
        # val_patient_acc instead of the raw per-epoch value. With a small
        # patient-level validation set, ONE patient flipping correct/incorrect
        # can swing raw accuracy by several points epoch-to-epoch -- smoothing
        # stops that single flip from being mistaken for real improvement.
        self.checkpoint_smoothing_window = config.get("checkpoint_smoothing_window", 3)

    def _smoothed_patient_acc(self) -> float:
        """Mean patient accuracy over the last `checkpoint_smoothing_window`
        epochs (expanding window until that many epochs exist yet)."""
        window = self.history["val_patient_acc"][-self.checkpoint_smoothing_window:]
        return sum(window) / len(window)

    def _run_epoch(self, loader: DataLoader, training: bool):
        self.student.train() if training else self.student.eval()
        total_loss = total_ce = total_kd = correct = total = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.no_grad():
                    teacher_logits = self.teacher(images)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    student_logits = self.student(images)
                    loss, ce_loss, kd_loss = self.criterion(
                        student_logits, teacher_logits, labels
                    )

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                bs = images.size(0)
                total_loss += loss.item() * bs
                total_ce += ce_loss.item() * bs
                total_kd += kd_loss.item() * bs
                preds = student_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += bs

        return total_loss / total, total_ce / total, total_kd / total, (correct / total)

    def _evaluate_student_patient_level(self, loader: DataLoader) -> float:
        self.student.eval()
        patient_probs_list = defaultdict(list)
        patient_label = {}

        with torch.no_grad():
            for batch in loader:
                images, labels, paths = batch[0], batch[1], batch[2]
                images = images.to(self.device, non_blocking=True)
                probs = F.softmax(self.student(images), dim=1).cpu()

                for label, p, p_path in zip(labels, probs, paths):
                    pid = _parse_patient_id(Path(p_path))
                    patient_probs_list[pid].append(p)
                    patient_label[pid] = label.item()

        correct = 0
        total_patients = len(patient_probs_list)
        if total_patients <= 1 and len(loader.dataset) > 1:
            print(f"[WARNING] Patient-level eval collapsed {len(loader.dataset)} images into "
                  f"{total_patients} 'patient' bucket(s) -- this is almost certainly a broken "
                  f"patient-ID parser, not real data. vl_patient_acc is not trustworthy.")

        for pid, list_of_probs in patient_probs_list.items():
            mean_prob = torch.stack(list_of_probs).mean(dim=0)
            if mean_prob.argmax().item() == patient_label[pid]:
                correct += 1

        return correct / total_patients if total_patients > 0 else 0.0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        epochs = self.config.get("epochs", 30)
        patience = self.config.get("patience", 10)
        no_improve = 0

        if self.config.get("use_class_weights", False) and self.num_classes:
            print("Computing class weights from train_loader...")
            self.class_weights = compute_class_weights(train_loader, self.num_classes, self.device)
            self.criterion = DistillationLoss(
                temperature=self.config.get("temperature", 4.0),
                alpha=self.config.get("alpha", 0.7),
                class_weights=self.class_weights,
                label_smoothing=self.config.get("label_smoothing", 0.1),
            )

        print(f"\n{'='*65}")
        print(f" Knowledge Distillation Training (Patient-Level Selection)")
        print(f" T={self.config.get('temperature', 4.0)} alpha={self.config.get('alpha', 0.7)} "
              f"lr_schedule={self.lr_schedule_type}")
        print(f"{'='*65}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_ce, tr_kd, tr_acc = self._run_epoch(train_loader, True)
            vl_loss, vl_ce, vl_kd, vl_acc = self._run_epoch(val_loader, False)
            
            vl_patient_acc = self._evaluate_student_patient_level(val_loader)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(vl_loss)
            self.history["val_acc"].append(vl_acc)
            self.history["val_patient_acc"].append(vl_patient_acc)

            smoothed_acc = self._smoothed_patient_acc()
            self.history["val_patient_acc_smoothed"].append(smoothed_acc)

            if self.lr_schedule_type == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(smoothed_acc)

            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Ep [{epoch:3d}/{epochs}] | "
                f"Train Loss: {tr_loss:.4f} | "
                f"Val Loss: {vl_loss:.4f} | "
                f"Val Acc(img): {vl_acc:.4f} | "
                f"Val Acc(patient) raw/smoothed: {vl_patient_acc:.4f} / {smoothed_acc:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            if smoothed_acc > self.best_val_patient_acc:
                self.best_val_patient_acc = smoothed_acc
                self._save("best_student.pt")
                no_improve = 0
                print(f"   Best SMOOTHED patient val acc: {smoothed_acc:.4f} "
                      f"(raw this epoch: {vl_patient_acc:.4f}) — saved")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break  
                
        hp = self.output_dir / "distill_history.json"
        with open(hp, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nDistillation complete. Best SMOOTHED patient val acc: {self.best_val_patient_acc:.4f}")
        return self.history

    def _save(self, name: str):
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "best_val_patient_acc": self.best_val_patient_acc,  # smoothed (k-epoch moving average), not a single raw epoch
            "config": self.config,
        }, self.output_dir / name)

    def load_best(self):
        ckpt = torch.load(self.output_dir / "best_student.pt", map_location=self.device)
        self.student.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best student (smoothed val_patient_acc={ckpt.get('best_val_patient_acc', 0.0):.4f})")


#_____________________________________________________________
# FALLBACK TRACK: Feature-level distillation
#_____________________________________________________________
class FeatureDistillationLoss(nn.Module):
    def __init__(self, feature_weight: float = 0.5, class_weights=None, label_smoothing: float = 0.1):
        super().__init__()
        self.feature_weight = feature_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, student_logits, student_feature_proj, teacher_feature, labels):
        ce_loss = self.ce(student_logits, labels)
        s = F.normalize(student_feature_proj, dim=-1)
        t = F.normalize(teacher_feature.detach(), dim=-1)
        feat_loss = F.mse_loss(s, t)
        total = (1.0 - self.feature_weight) * ce_loss + self.feature_weight * feat_loss
        return total, ce_loss, feat_loss


class FeatureDistillationTrainer(DistillationTrainer):
    """
    Fallback distillation track: same overall training loop/checkpointing/
    early-stopping/patient-level selection as DistillationTrainer, but the
    KD signal is teacher-student FEATURE similarity instead of teacher-
    student LOGIT similarity. A learnable linear projection maps the
    student's (smaller) feature dimension into the teacher's feature space
    before comparing them.

    Requires the teacher to support forward(x, return_feature=True) (true
    of ViTfBCD) and the student to be a timm-style model exposing
    forward_features()/forward_head() (true of the deit_tiny/deit_small
    students built by build_student()). A different student architecture
    would need its own feature-extraction hook wired in here.
    """
    def __init__(self, teacher, student, config, device,
                 output_dir="/home/user/Proj-Ploy/vit_breast_cancer/outputs/distill",
                 num_classes=None):
        super().__init__(teacher, student, config, device, output_dir, num_classes)

        teacher_hidden_dim = self.teacher.vit.heads.head[0].in_features
        student_embed_dim = getattr(self.student, "embed_dim", None)
        if student_embed_dim is None:
            raise AttributeError(
                "Student model has no `.embed_dim` attribute -- feature-level distillation "
                "needs the student's feature dimensionality to build the projection layer. "
                "timm ViT/DeiT students expose this; a different architecture would need "
                "this wired in explicitly."
            )
        self.feature_projection = nn.Linear(student_embed_dim, teacher_hidden_dim).to(device)

        # Rebuild the optimizer to include the projection layer's params
        # (created after the parent __init__ already built one without them).
        self.optimizer = AdamW(
            list(self.student.parameters()) + list(self.feature_projection.parameters()),
            lr=config.get("lr", 5e-5),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        if self.lr_schedule_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.get("epochs", 30), eta_min=config.get("min_lr", 1e-6))
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=config.get("lr_factor", 0.5),
                patience=config.get("lr_patience", 3), min_lr=config.get("min_lr", 1e-6))

        self.criterion = FeatureDistillationLoss(
            feature_weight=config.get("feature_weight", 0.5),
            class_weights=None,
            label_smoothing=config.get("label_smoothing", 0.1),
        )

    def _student_forward_with_feature(self, images):
        feat = self.student.forward_features(images)
        if feat.dim() == 3:
            feat = feat[:, 0]  # CLS token
        logits = self.student.forward_head(feat) if hasattr(self.student, "forward_head") else self.student(images)
        return logits, feat

    def _run_epoch(self, loader: DataLoader, training: bool):
        self.student.train() if training else self.student.eval()
        total_loss = total_ce = total_feat = correct = total = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.no_grad():
                    _teacher_logits, teacher_feature = self.teacher(images, return_feature=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    student_logits, student_feature = self._student_forward_with_feature(images)
                    student_feature_proj = self.feature_projection(student_feature)
                    loss, ce_loss, feat_loss = self.criterion(
                        student_logits, student_feature_proj, teacher_feature, labels)

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.student.parameters()) + list(self.feature_projection.parameters()), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                bs = images.size(0)
                total_loss += loss.item() * bs
                total_ce += ce_loss.item() * bs
                total_feat += feat_loss.item() * bs
                preds = student_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += bs

        return total_loss / total, total_ce / total, total_feat / total, (correct / total)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        # Copy of DistillationTrainer.fit(), with the one incompatible line
        # fixed: the parent version rebuilds self.criterion as a plain
        # DistillationLoss() when use_class_weights is set, which would
        # silently replace this fallback's FeatureDistillationLoss.
        epochs = self.config.get("epochs", 30)
        patience = self.config.get("patience", 10)
        no_improve = 0

        if self.config.get("use_class_weights", False) and self.num_classes:
            print("Computing class weights from train_loader...")
            self.class_weights = compute_class_weights(train_loader, self.num_classes, self.device)
            self.criterion = FeatureDistillationLoss(
                feature_weight=self.config.get("feature_weight", 0.5),
                class_weights=self.class_weights,
                label_smoothing=self.config.get("label_smoothing", 0.1),
            )

        print(f"\n{'='*65}")
        print(" FALLBACK: Feature-Level Distillation Training (Patient-Level Selection)")
        print(f" feature_weight={self.config.get('feature_weight', 0.5)} lr_schedule={self.lr_schedule_type}")
        print(f"{'='*65}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_ce, tr_feat, tr_acc = self._run_epoch(train_loader, True)
            vl_loss, vl_ce, vl_feat, vl_acc = self._run_epoch(val_loader, False)

            vl_patient_acc = self._evaluate_student_patient_level(val_loader)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(vl_loss)
            self.history["val_acc"].append(vl_acc)
            self.history["val_patient_acc"].append(vl_patient_acc)

            smoothed_acc = self._smoothed_patient_acc()
            self.history["val_patient_acc_smoothed"].append(smoothed_acc)

            if self.lr_schedule_type == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(smoothed_acc)

            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Ep [{epoch:3d}/{epochs}] | "
                f"Train Loss: {tr_loss:.4f} (ce={tr_ce:.3f}|feat={tr_feat:.3f}) | "
                f"Val Loss: {vl_loss:.4f} | "
                f"Val Acc(img): {vl_acc:.4f} | "
                f"Val Acc(patient) raw/smoothed: {vl_patient_acc:.4f} / {smoothed_acc:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            if smoothed_acc > self.best_val_patient_acc:
                self.best_val_patient_acc = smoothed_acc
                self._save("best_student.pt")
                no_improve = 0
                print(f"   Best SMOOTHED patient val acc: {smoothed_acc:.4f} "
                      f"(raw this epoch: {vl_patient_acc:.4f}) — saved")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        hp = self.output_dir / "distill_history_feature_fallback.json"
        with open(hp, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nFeature-level distillation complete. Best SMOOTHED patient val acc: {self.best_val_patient_acc:.4f}")
        return self.history


def check_collapse(history: dict, min_loss_decrease: float = 0.10,
                    collapse_acc_threshold: float = 0.65) -> dict:
    reasons = []
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    val_patient_acc = history.get("val_patient_acc", [])

    def has_nan_inf(values):
        return any(v != v or v in (float("inf"), float("-inf")) for v in values)

    if has_nan_inf(train_loss) or has_nan_inf(val_loss):
        reasons.append("NaN/Inf detected in train_loss or val_loss -- numerical collapse "
                       "(check learning rate, gradient clipping, or mixed-precision scaling).")

    if len(train_loss) >= 2:
        start, end = train_loss[0], train_loss[-1]
        rel_decrease = (start - end) / start if start > 0 else 0.0
        if rel_decrease < min_loss_decrease:
            reasons.append(
                f"train_loss barely moved: {start:.4f} -> {end:.4f} "
                f"({rel_decrease*100:.1f}% relative decrease, threshold is "
                f"{min_loss_decrease*100:.0f}%) -- the optimizer doesn't appear to be "
                "fitting anything."
            )

    if val_acc and max(val_acc) < collapse_acc_threshold:
        reasons.append(
            f"val_acc (image-level) never exceeded {max(val_acc):.4f} at any point in "
            f"{len(val_acc)} epochs (threshold: {collapse_acc_threshold:.2f}) -- the student "
            "never learned to separate classes at all, even briefly."
        )

    if val_patient_acc:
        unique_vals = set(round(v, 6) for v in val_patient_acc)
        if len(unique_vals) == 1 and next(iter(unique_vals)) < collapse_acc_threshold:
            reasons.append(
                f"val_patient_acc is STUCK at a single value ({next(iter(unique_vals)):.4f}) "
                f"across all {len(val_patient_acc)} epochs, below the collapse threshold -- "
                "looks like a degenerate 'always predicts the same class' pattern."
            )

    return {"collapsed": len(reasons) > 0, "reasons": reasons}


#_____________________________________________________________
# Quantization Block
#_____________________________________________________________
def quantize_model_int8(
    model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    output_path: str = "/home/user/Proj-Ploy/vit_breast_cancer/outputs/distill/efficientnet_b0_int8.pt",
    is_transformer: bool = False,
) -> nn.Module:
    print(f"Applying INT8 Quantization (Mode: {'Dynamic' if is_transformer else 'Static'})...")
    model_cpu = model.to("cpu").eval()

    if is_transformer:
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    else:
        model_cpu.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.ao.quantization.prepare(model_cpu, inplace=False)
        
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= 10: 
                    break
                # 💡 แก้ไขจุดส่งภาพสำหรับ Static Quantization
                images = batch[0]
                model_prepared(images.to("cpu"))
                
        quantized_model = torch.ao.quantization.convert(model_prepared, inplace=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    print(f" INT8 model saved -> {output_path}")

    return quantized_model


def compare_model_sizes(
    fp32_model: nn.Module, 
    int8_model: nn.Module,
    fp32_path: str, 
    int_path: str,
    fp32_acc: float = None,  
    int8_acc: float = None,
    teacher_path: str = None,
    teacher_acc: float = None,
    student_name: str = "student",
    output_dir: str = None,
):
    fp32_mb = os.path.getsize(fp32_path) / 1e6 if os.path.exists(fp32_path) else 0.0
    int8_mb = os.path.getsize(int_path) / 1e6 if os.path.exists(int_path) else 0.0
    fp32_p = sum(p.numel() for p in fp32_model.parameters()) / 1e6

    teacher_mb = None
    if teacher_path and os.path.exists(teacher_path):
        teacher_mb = os.path.getsize(teacher_path) / 1e6

    print("\n====================================================")
    print(" Model Compression & Accuracy Summary")
    print("====================================================")
    print(f" {'Model':<16} | {'Params (M)':<10} | {'File (MB)':<10} | {'Val Acc':<10}")
    print("-" * 52)

    if teacher_mb is not None:
        teacher_acc_str = f"{teacher_acc*100:.2f}%" if teacher_acc is not None else "N/A"
        print(f" {'Teacher (FP32)':<16} | {'--':<10} | {teacher_mb:<10.1f} | {teacher_acc_str:<10}")

    fp32_acc_str = f"{fp32_acc*100:.2f}%" if fp32_acc is not None else "N/A"
    print(f" {'FP32 Student':<16} | {fp32_p:<10.2f} | {fp32_mb:<10.1f} | {fp32_acc_str:<10}")

    if int8_mb:
        reduction = (1 - int8_mb / fp32_mb) * 100 if fp32_mb else 0
        int8_acc_str = f"{int8_acc*100:.2f}%" if int8_acc is not None else "N/A"
        size_str = f"{int8_mb:.1f} (-{reduction:.0f}%)"
        print(f" {'INT8 Quantized':<16} | {fp32_p:<10.2f} | {size_str:<10} | {int8_acc_str:<10}")
    print("====================================================")

    if output_dir:
        summary = {
            "student_name": student_name,
            "teacher": {"size_mb": teacher_mb, "val_acc": teacher_acc},
            "student_fp32": {"size_mb": fp32_mb, "params_m": fp32_p, "val_acc": fp32_acc},
            "student_int8": {"size_mb": int8_mb, "val_acc": int8_acc},
        }
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "compression_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Saved -> {out_path}")

def evaluate_teacher(teacher: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    teacher = teacher.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) 
            
            preds = teacher(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            
    acc = correct / total if total > 0 else 0.0
    print(f"[Teacher standalone] Val Acc: {acc:.4f}")
    return acc


def evaluate_quantized_model(model: nn.Module, data_loader: DataLoader) -> float:
    torch.backends.quantized.engine = "fbgemm"
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0], batch[1]
            images = images.to("cpu")
            labels = labels.to("cpu")
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += images.size(0)
            
    return correct / total if total > 0 else 0.0