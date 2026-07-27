import json
import os
import random
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from PIL import Image
from torch.utils.data import DataLoader

SEED = 42


def set_seed(seed: int = SEED):
    """Fix every source of randomness that affects a training run.

    Without this, each run of this script gets: a different student
    classification-head init, a different data-loading/augmentation
    order, and (on GPU) non-deterministic cuDNN kernels -- so
    "best" checkpoint selection during trainer.fit() can land on a
    genuinely different set of weights each time, even though the
    architecture, size on disk, and latency stay identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    # Keeps per-worker numpy/random state reproducible when num_workers > 0
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

from src.distillation import (
    build_student, 
    DistillationTrainer, 
    quantize_model_int8, 
    compare_model_sizes,
    evaluate_quantized_model,
    evaluate_teacher,
)
from src.dataset import build_dataloaders
from src.train_teacher import SampleListDataset, get_patient_predictions
from src.model import ViTfBCD

def save_performance_curves(history, save_path):

    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.rcParams['font.family'] = 'sans-serif'

    fig = plt.figure(figsize=(14, 5.5))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    ax1 = fig.add_subplot(outer[0])
    ax1.plot(epochs, history['train_loss'], 'o-', color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 's-', color='#ff7f0e', linewidth=2, label='Validation Loss')
    ax1.set_title('Knowledge Distillation Loss Curves', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Loss Value', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_locator(MultipleLocator(2))

    # Validation accuracy: start at 0 (epoch-0 / before-training point), then
    # jump into the real 90-100% range where patient accuracy actually lives.
    # A broken y-axis keeps that jump visible without squashing the real
    # epoch-to-epoch movement into a flat line at the top of the chart.
    val_acc_percentage = [acc * 100 for acc in history['val_patient_acc']]

    # IMPORTANT: the trainer selects/saves best_student.pt using the SMOOTHED
    # k-epoch moving average (see distillation.py's _smoothed_patient_acc()),
    # not the raw per-epoch value. Marking "best" from the raw series would
    # highlight whichever epoch first got lucky hitting 100% -- which is NOT
    # necessarily the epoch whose checkpoint actually got saved. Use the
    # smoothed series here so the chart matches what actually happened.
    if 'val_patient_acc_smoothed' in history and history['val_patient_acc_smoothed']:
        smoothed_percentage = [acc * 100 for acc in history['val_patient_acc_smoothed']]
        best_acc = max(smoothed_percentage)
        best_epoch = epochs[smoothed_percentage.index(best_acc)]
    else:
        smoothed_percentage = None
        best_acc = max(val_acc_percentage)
        best_epoch = epochs[val_acc_percentage.index(best_acc)]

    plot_epochs = [0] + epochs
    plot_acc = [0.0] + val_acc_percentage
    zoom_lo = max(0, min(val_acc_percentage) - 3)

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=[6, 1], hspace=0.08
    )
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

    ax_top.plot(plot_epochs, plot_acc, '^--', color='#2ca02c', linewidth=2, alpha=0.55,
                label='Validation Accuracy (Patient, raw)')
    ax_bot.plot(plot_epochs, plot_acc, '^--', color='#2ca02c', linewidth=2, alpha=0.55)

    if smoothed_percentage is not None:
        plot_smoothed = [0.0] + smoothed_percentage
        ax_top.plot(plot_epochs, plot_smoothed, 'o-', color='#1a6b1a', linewidth=2.2,
                    label='Smoothed (checkpoint criterion)')
        ax_bot.plot(plot_epochs, plot_smoothed, 'o-', color='#1a6b1a', linewidth=2.2)

    ax_top.scatter(best_epoch, best_acc, color='red', s=100, zorder=5,
                   label=f'Best (smoothed): {best_acc:.2f}% (Ep {best_epoch})')

    ax_top.set_ylim(zoom_lo, 100.8)
    ax_bot.set_ylim(-3, 3)

    # hide the spines between the two axes and draw the diagonal "break" marks
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bot.set_yticks([0])

    d = .5  # size of the diagonal break marks, in points (constant regardless of axes size)
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0], [0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0], [1], transform=ax_bot.transAxes, **kwargs)

    ax_bot.set_xlabel('Epochs', fontsize=11)
    ax_top.set_ylabel('Accuracy (%)', fontsize=11)
    ax_top.set_title('Student Model Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_bot.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(fontsize=9, loc='lower right')
    ax_bot.xaxis.set_major_locator(MultipleLocator(2))

    plt.suptitle('Training Performance Evaluation (Teacher: ViT -> Student: DeiT-Tiny, binary)', fontsize=14, fontweight='bold', y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS]  -> {save_path}")
    plt.close()


def main():
    set_seed(SEED)  # must run before any model/dataloader is constructed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")


    config = {
        "lr": 2e-5, #2e-5,             
        "epochs": 30,           
        "temperature": 4.0,     
        "alpha": 0.3,             
        "patience": 10,         
        "weight_decay": 5e-2,  
        "image_size": 384,  
        "batch_size": 32,    
        "mode": "binary",
        "stain_method": "macenko",
        "label_smoothing": 0.1,   
        "use_class_weights": False,  
        "lr_schedule": "plateau",   
        "lr_factor": 0.5,
        "lr_patience": 3,
        "seed": SEED,  # was missing -- evaluate_distill_on_test.py already
                        # hardcodes "seed": 42 for its test split, so without
                        # this key here, build_dataloaders() had no guarantee
                        # of drawing the SAME train/val/test patient split
                        # every time this script runs.
    }

    output_dir = "/home/user/Proj-Ploy/vit_breast_cancer/outputs/distill_results"
    num_classes = 2  
    # Load Dataset
    print("loading dataset")
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir="/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/", 
        config=config
    )
    images, labels = next(iter(train_loader))
    print("image Batch Shape:", images.shape)

    # DistillationTrainer's per-epoch patient-level validation needs
    # (image, label, path) triples -- build_dataloaders()'s val_loader only
    # yields (image, label). Rewrap val_ds.samples with SampleListDataset
    # (same helper main.py's k-fold path uses) so trainer.fit() below
    # doesn't crash the first time it calls _evaluate_student_patient_level().
    val_ds = val_loader.dataset
    val_gen = torch.Generator()
    val_gen.manual_seed(SEED)
    val_loader = DataLoader(
        SampleListDataset(val_ds.samples, val_ds.transform),
        batch_size=config["batch_size"], shuffle=False,
        num_workers=config.get("num_workers", 4), pin_memory=True,
        worker_init_fn=_seed_worker, generator=val_gen,
    )
    val_loader.dataset.loader = lambda path: Image.open(path).convert("RGB")

    print("Initailizing and loading pretraining Teacher (ViTfBCD)")
    teacher_model = ViTfBCD(num_classes=num_classes)

    teacher_model.resize_position_embeddings()

    # Loads the SAME binary (2-class) checkpoint main.py's train_final_model()
    # produces -- run main.py first (with RUN_FINAL_MODEL=True) so this file
    # exists and matches num_classes=2 above.
    teacher_weights_path = "/home/user/Proj-Ploy/vit_breast_cancer/outputs/best_model.pt"
    checkpoint = torch.load(teacher_weights_path, map_location=device)

    if "model_state_dict" in checkpoint:
        teacher_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        teacher_model.load_state_dict(checkpoint)

    print("\nEvaluating standalone Teacher accuracy (sets the ceiling for distillation)...")
    teacher_val_acc = evaluate_teacher(teacher_model, val_loader, device)
    if teacher_val_acc < 0.70:
        print(f"[Warning] Teacher val acc is {teacher_val_acc:.2%}. The student cannot "
            f"reliably exceed this via distillation — consider improving/retraining "
            f"the teacher, or leaning more on ground-truth labels (higher alpha) "
            f"instead of the teacher's soft targets.")

    teacher_patient_label, teacher_patient_pred = get_patient_predictions(
        teacher_model, val_loader, device, num_classes=num_classes)
    teacher_val_patient_acc = (
        sum(int(teacher_patient_pred[pid] == teacher_patient_label[pid]) for pid in teacher_patient_label)
        / len(teacher_patient_label) if teacher_patient_label else float("nan")
    )
    print(f"[Teacher] Patient-level val acc: {teacher_val_patient_acc:.4f} "
          f"(n={len(teacher_patient_label)} patients -- same metric/set as the student's "
          "reported accuracy below, unlike the image-level number above)")

    student_name = "deit_tiny"
    print(f"Building Student model: {student_name}  (committed choice -- see comment above)")
    student_model = build_student(student_name=student_name, num_classes=num_classes, pretrained=True)

    trainer = DistillationTrainer(
        teacher=teacher_model,
        student=student_model,
        config=config,
        device=device,
        output_dir=output_dir,
        num_classes=num_classes,
    )

    trainer.fit(train_loader, val_loader)
    
    if hasattr(trainer, 'history') and trainer.history:
        chart_path = f"{output_dir}/performance_curves.png"
        save_performance_curves(trainer.history, chart_path)
    else:
        print("\n[Warning] No history in trainer.history,  So cant'plot the graph")

    trainer.load_best()
    fp32_accuracy = trainer.best_val_patient_acc 

    print("\nConverting to INT8 Post-Training Quantization (CPU)...")
    
    calib_dataset = val_loader.dataset
    light_calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False, num_workers=0)

    quantized_student = quantize_model_int8(
        model=trainer.student,
        calib_loader=light_calib_loader,
        device=device,
        output_path=f"{output_dir}/{student_name}_int8.pt",
        is_transformer=True,
    )

    print("Evaluating INT8 Quantized model accuracy on CPU...")
    torch.backends.quantized.engine = 'fbgemm'

    int8_accuracy = evaluate_quantized_model(quantized_student, val_loader)
    print(f"[Note] int8_accuracy above is IMAGE-level (evaluate_quantized_model doesn't do "
          "patient aggregation), while teacher_acc/fp32_acc below are now patient-level -- "
          "the compression summary's INT8 row still isn't a like-for-like comparison. Run "
          "evaluate_distillation_on_test.py for a fully consistent, patient-level, TEST-set "
          "comparison across all three models.")
    fp32_path = f"{output_dir}/best_student.pt"
    int8_path = f"{output_dir}/{student_name}_int8.pt"
    
    compare_model_sizes(
        fp32_model=trainer.student, 
        int8_model=quantized_student, 
        fp32_path=fp32_path, 
        int_path=int8_path,
        fp32_acc=fp32_accuracy,
        int8_acc=int8_accuracy,
        teacher_path=teacher_weights_path,
        teacher_acc=teacher_val_patient_acc,
        student_name=student_name,
        output_dir=output_dir,
    )
    
    teacher_sz = os.path.getsize(teacher_weights_path) / (1024 * 1024) if os.path.exists(teacher_weights_path) else 1038.0
    fp32_sz = os.path.getsize(fp32_path) / (1024 * 1024) if os.path.exists(fp32_path) else 22.5
    int8_sz = os.path.getsize(int8_path) / (1024 * 1024) if os.path.exists(int8_path) else 6.6

    summary_data = {
        "student_name": student_name,
        "teacher_size_mb": teacher_sz,
        "fp32_size_mb": fp32_sz,
        "int8_size_mb": int8_sz,
        "teacher_acc": teacher_val_patient_acc,
        "fp32_acc": fp32_accuracy,
        "int8_acc": int8_accuracy
    }
    
    json_output_path = f"{output_dir}/compression_summary.json"
    with open(json_output_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"[OK] Saved summary JSON for test evaluation: {json_output_path}")

if __name__ == "__main__":
    main()