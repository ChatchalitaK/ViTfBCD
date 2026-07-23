import json
import matplotlib.pyplot as plt
from PIL import Image

# ── Load training history ───────────────────────────────────────────────────
with open("/home/user/Proj-Ploy/vit_breast_cancer/outputs/history.json", "r") as f:
    history = json .load(f)

# ── Plot training curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_title("Loss"); axes[0].legend()


axes[1].plot(history["train_acc"], label="Train Acc")
axes[1].plot(history["val_acc"], label="Val Acc")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout()
plt.show()

# ── Load and display matrics ───────────────────────────────────────────────────
with open("../outputs/vitfbcd_base_binary_40x/test_matrics.json") as f:
    matrics = json.load(f)
print(matrics)

# ── Single attention visualization ─────────────────────────────────────────────
image = Image.open("../outputs/vitfbcd_base_binary_40x/confusion_matrix.png")
image.show()