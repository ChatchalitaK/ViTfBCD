import json
from pathlib import Path
import numpy as np

OUTPUTS_DIR = Path("/home/user/Proj-Ploy/vit_breast_cancer/outputs")
SUMMARY_OUT_PATH = OUTPUTS_DIR / "kfold_summary.json"

fold_accuracies = []

for fold_num in range(1, 6):
    fold_dir = OUTPUTS_DIR / f"fold_{fold_num}"
    history_file = fold_dir / "history.json"
    
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
                
            val_accs = (
                history.get("val_patient_acc") or 
                history.get("val_acc") or 
                history.get("val_accuracy") or 
                history.get("val_accs")
            )
            
            if val_accs and isinstance(val_accs, list):
                best_acc = max(val_accs)
                fold_accuracies.append(float(best_acc))
                print(f"[Fold {fold_num}] Best Val Accuracy: {best_acc:.4f} ({history_file.name})")
            else:
                print(f"[WARNING] No valid validation accuracy list found in {history_file}")
        except Exception as e:
            print(f"[ERROR] Failed to read {history_file}: {e}")
    else:
        print(f"[WARNING] File not found: {history_file}")

if fold_accuracies:
    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    
    summary_data = {
        "k_folds": len(fold_accuracies),
        "mean_patient_accuracy": mean_acc,
        "std_patient_accuracy": std_acc,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "fold_accuracies": fold_accuracies
    }
    
    with open(SUMMARY_OUT_PATH, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print("\n" + "=" * 50)
    print(f"[SUCCESS] Updated {SUMMARY_OUT_PATH}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print("=" * 50)
else:
    print("\n[ERROR] Could not extract validation accuracies from fold_1..5 directories.")
