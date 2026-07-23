## Knowledge Distillation & Model Compression

### Methodology: Knowledge Distillation Theory and Parameters

**Logit-level distillation (primary track).** Following Hinton et al.
(2015), the student is trained against a weighted combination of two
signals: the standard hard-label cross-entropy loss against the ground
truth, and a soft-target loss against the teacher's *softened* output
distribution. Given teacher logits `z_t` and student logits `z_s`, both are
divided by a temperature `T` before softmax:

```
p_t = softmax(z_t / T),   p_s = softmax(z_s / T)
L_KD = KL(p_t || p_s) * T^2
L_CE = CrossEntropy(z_s, y)
L_total = alpha * L_CE + (1 - alpha) * L_KD
```

Raising `T` above 1 flattens the teacher's softmax, revealing the relative
probabilities the teacher assigns to the *non-target* classes -- Hinton's
"dark knowledge" -- which carries more information than the one-hot label
alone (e.g. how confidently the teacher distinguishes a benign case from a
*specific kind* of malignant case, not just "not malignant"). The `T^2`
scaling factor corrects for the gradient magnitude shrinking as `T`
increases, keeping the KD loss's contribution comparable across
temperature settings.

**Parameters used in this project** (`run_distill.py`'s config, printed at
the start of every training run):

| Parameter | Value | Rationale |
|---|---|---|
| Temperature (`T`) | 4.0 | Moderate softening -- high enough to expose inter-class structure beyond the hard label, not so high that the teacher's distribution degrades toward uniform noise. |
| `alpha` | 0.3 | Weights hard-label supervision at 30%, soft-target supervision at 70% -- leans toward trusting the teacher's distribution, appropriate given the teacher's own validated accuracy (see Table 2). |
| LR schedule | `plateau` (`ReduceLROnPlateau`) | Reduces LR when patient-level validation accuracy stalls, rather than a fixed cosine decay -- matches the primary model's own training recipe (see `main.py`). |
| Backbone freezing | Layer-wise LR via `get_layered_parameters()` | Consistent with the primary model's fine-tuning approach; prevents the distillation signal from catastrophically disturbing pretrained low-level features. |
| Checkpoint selection | `val_acc` (image-level, primary) | **Not** `val_patient_acc` -- see the Limitations note below for why the coarse patient-level metric was found unsuitable as a selection criterion. |

**Feature-level distillation (fallback track).** If the logit-level run
shows signs of collapse -- checked automatically via `check_collapse()`
immediately after each `run_distill.py` run (NaN/Inf losses, `train_loss`
not decreasing, or `val_acc`/`val_patient_acc` never rising above a
near-chance threshold) -- the pipeline switches, without any further
alpha/temperature tuning, to matching CLS-token *features* between teacher
and student instead of matching output logits (`FeatureDistillationLoss`,
`FeatureDistillationTrainer`). A learned linear projection maps the
student's smaller embedding dimension into the teacher's feature space,
and the two are compared via L2-normalized MSE. This is a fundamentally
different supervisory signal -- what the model *represents* internally
rather than only its final decision -- which can succeed in exactly the
case where logit matching fails to provide useful gradient (e.g. a teacher
whose softened outputs are already close to one-hot, or too large a
capacity gap between teacher and student for logit-only supervision to
transfer well).

**Student architecture: DeiT-Tiny.** Selected once, over MobileNetV3, and
not revisited (see `run_distill.py`'s inline rationale): logit-level KD
transfers best when teacher and student share the same function class
(both patch/attention-based transformers), DeiT was designed around
exactly this "distill from a larger ViT teacher" recipe, and the INT8
quantization path below is already correctly wired for a transformer
student (dynamic quantization of `nn.Linear` layers) without the
additional QuantStub/DeQuantStub work a CNN student would require.

**Post-training INT8 quantization.** After the chosen track (logit-level
or feature-level fallback) converges, `quantize_model_int8(...,
is_transformer=True)` applies dynamic quantization to the student's
`nn.Linear` layers -- weights are quantized to INT8, activations remain
FP32 and are quantized on the fly per inference call. This targets
edge/CPU deployment specifically (dynamic quantization's benefit is
primarily size and CPU inference latency, not GPU throughput).

### Table 1: Model Efficiency Comparison

| Model | Params (M) | Size (MB) | Compression vs. Teacher | Test Patient Accuracy | Inference Latency (ms/image) |
|---|---|---|---|---|---|
| Teacher (ViTfBCD, FP32) | 86.49 | 1038.0 | -- (baseline) | 94.12% | 110.83 |
| Student (deit_tiny, FP32) | 5.60 | 22.5 | 46.1x | 94.12% | 14.59 |
| Student (deit_tiny, INT8) | 5.60 | 6.6 | 157.3x | 94.12% | 15.16 |

*(Auto-filled by `fill_distillation_report.py` from
`compression_summary.json`, `distillation_test_set_comparison.json`, and a
live latency benchmark -- see `efficiency_section.md` for the original
version of this table and its measurement methodology in full.)*

### Table 2: Accuracy Across Evaluation Protocols

| Model | Evaluation | n | Accuracy |
|---|---|---|---|
| Teacher | 5-fold CV (patient-level, mean ± std) | 65 patients (pool) | 87.28% ± 7.60% |
| Teacher (Baseline) | Held-out TEST (patient-level) | 17 patients | 94.12% |
| Student FP32 | Held-out TEST (patient-level) | 17 patients | 94.12% |
| Student INT8 | Held-out TEST (patient-level) | 17 patients | 94.12% |

*(Auto-filled by `fill_distillation_report.py`)*

### Figure 1: Model Size Compression Comparison

`outputs/model_compression_chart.png` (generated by `plot_compression.py`)
-- log-scale bar chart of teacher/FP32-student/INT8-student file sizes,
annotated with percentage size reduction at each stage.

### Figure 2: Knowledge Distillation Training Curves

`outputs/distill_results/performance_curves.png` (generated by
`save_performance_curves()` in `run_distill.py`) -- train/validation loss
curves and student validation accuracy across training epochs.

### Study Limitations / Methods Note

- **Small validation and test sets.** The distillation validation set has
  only 12 patients and the held-out test set only 17 -- both far smaller
  than typical deep learning validation practice. Patient-level accuracy
  on 12 patients has only 13 possible values (0/12 through 12/12), which
  was found to make it an unreliable *checkpoint-selection* criterion: an
  early, undertrained epoch reached the ceiling value (12/12) purely by
  chance and would have been kept as "best" for the rest of a 10-epoch
  patience window despite train/validation loss continuing to improve
  substantially afterward. Checkpoint selection was changed to the
  finer-grained, continuous image-level `val_acc` as a result; patient-level
  accuracy is still reported (it is the clinically relevant unit) but no
  longer gates which checkpoint is kept.
- **Single-split distillation training.** Unlike the primary teacher model
  (evaluated via 5-fold patient-wise cross-validation), the distillation
  runs reported here use a single train/val/test split. The 5-fold CV
  number in Table 2 characterizes the *teacher's* variance across splits
  only; the student's stability across different splits/seeds has not
  been separately characterized. If time permits, repeating the
  distillation run across multiple seeds (the same pattern used for the
  BACH fine-tuning cross-validation) would give a mean ± std for the
  student's accuracy rather than a single-run point estimate.
- **INT8 quantization is CPU-only and dynamic (Linear-layer-only).**
  Reported latency figures reflect single-image (batch size 1) CPU
  inference; they are not representative of GPU throughput or
  larger-batch serving scenarios.
- **BACH/PCam cross-dataset generalization was evaluated on the primary
  (teacher) model only** (see `cross_validation_section.md`), not on the
  distilled student -- whether the compressed student generalizes as well
  as the teacher to external datasets is untested and should not be
  assumed from these results alone.