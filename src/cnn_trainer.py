"""
cnn_trainer.py - Training loop for CNN baseline models
Handles InceptionV3' aux_logits during trainig.
Same hyperparemeter structure as ViTfBCD trainer for fair camparison.
"""


import time
import json
from pathlib import Path  
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  

from src.cnn_models import InceptionV3Baseline

class CNNTrainer:
    """
    Training pipeline for CNN baseline model.
    Identical hyperparameters to ViTfBCD Trainer for fair comparison.
    """

    def __init__(self, model, model_name: str, config: dict, device: torch.device, output_dir: str):
        self.model = model.to(device)
        self.model_name = model_name
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)  
        self.output_dir.mkdir(parents=True, exist_ok=True)  

        self.is_inception = isinstance(model, InceptionV3Baseline)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
        
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.get("lr", 1e-4), 
            weight_decay=config.get("weight_decay", 1e-4)
        )
        
        self.lr_schedule_type = config.get("lr_schedule", "cosine")
        if self.lr_schedule_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # monitors val_acc, which we want to maximize
                factor=config.get("lr_factor", 0.5),
                patience=config.get("lr_patience", 3),
                min_lr=config.get("min_lr", 1e-6),
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("epochs", 30),
                eta_min=config.get("min_lr", 1e-6)
            )
        
        # Mixed precision setup
        self.scaler = GradScaler()
        self.epochs = config.get("epochs", 30)
        
        # History tracking
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type):
                if self.is_inception:
                    # InceptionV3 returns (main_output, aux_output) during training
                    outputs, aux_outputs = self.model(images)
                    loss_main = self.criterion(outputs, labels)
                    loss_aux = self.criterion(aux_outputs, labels)
                    # Standard weight for Inception auxiliary classifier is 0.3
                    loss = loss_main + 0.3 * loss_aux
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            
            # Backward pass & Optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast(device_type=self.device.type):
                    # InceptionV3 only returns main output during evaluation
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"[*] Starting training for {self.model_name}...")
        best_val_acc = 0.0
        
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            if self.lr_schedule_type == "plateau":
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()
            
            # Record metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{self.epochs}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
            
            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint("best_model.pth")
                
        # Save final metrics history
        self.save_history()
        print(f"[+] Training complete. Best Val Acc: {best_val_acc*100:.2f}%")

    def save_checkpoint(self, filename: str):
        checkpoint_path = self.output_dir / f"{self.model_name}_{filename}"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)

    def save_history(self):
        history_path = self.output_dir / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)