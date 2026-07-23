import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.train_teacher import make_patient_folds, SampleListDataset

from src.cnn_models import build_cnn_model, CNN_INPUT_SIZES
from src.cnn_trainer import CNNTrainer

def main():
    config = {
        "magnification": "all",
        "mode": "binary",
        "num_classes": 2,
        "batch_size": 32,
        "epochs": 30,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 5
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base_dir = Path("/home/user/Proj-Ploy/vit_breast_cancer/outputs/cnn_baselines")

    models_to_benchmark = ["resnet50", "inceptionv3", "irv2", "efficientnet_b4"]

    for model_name in models_to_benchmark:
        print(f"\n=============================================")
        print(f" Starting Benchmark for: {model_name.upper()}")
        print(f"=============================================")
        
        config["image_size"] = CNN_INPUT_SIZES.get(model_name, 384)
        model_output_dir = output_base_dir / model_name

        model = build_cnn_model(name=model_name, num_classes=config["num_classes"], pretrained=True)

        DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
        train_loader, val_loader, _ = build_dataloaders(DATA_DIR, config)

        trainer = CNNTrainer(
            model=model,
            model_name=model_name,
            config=config,
            device=device,
            output_dir=str(model_output_dir)
        )
        
        print(f"Training {model_name}...")
        trainer.fit(train_loader, val_loader)
        print(f" Saved metrics for {model_name} in {model_output_dir}")

if __name__ == "__main__":
    main()