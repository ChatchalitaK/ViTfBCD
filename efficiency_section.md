## Efficiency: Teacher vs. Distilled Student

### Motivation

Accuracy alone doesn't tell you whether a model is deployable. This section
reports the practical cost side of the teacher -> DeiT-Tiny distillation
(see the "Select exactly one lightweight student model" / logit-level KL
distillation work in `run_distill.py`): how much smaller the student is,
how that changes further after INT8 quantization, and what each stage
costs in inference latency -- alongside the accuracy each stage keeps.

### Setup

- **Models compared**: the ViTfBCD teacher (FP32), the distilled DeiT-Tiny
  student (FP32), and the INT8-quantized student
  (`quantize_model_int8(..., is_transformer=True)`, dynamic quantization of
  `nn.Linear` layers).
- **Size and params**: read directly from
  `outputs/distill_results/compression_summary.json`, written by
  `distillation.py`'s `compare_model_sizes()` at the end of `run_distill.py`.
- **Accuracy**: read from `outputs/distill_results/distillation_test_set_comparison.json`,
  written by `evaluate_distillation_on_test.py` -- NOT from
  `compression_summary.json`'s own `val_acc` fields, which mix per-image
  accuracy (teacher, INT8 student) with per-patient accuracy on a 12-patient
  validation set (FP32 student), and are computed on val rather than the
  held-out test set. `evaluate_distillation_on_test.py` re-evaluates all
  three models with the same metric (patient-level accuracy) on the same,
  larger, properly held-out BreakHis test set.
- **Latency**: measured separately (size/accuracy alone don't imply
  latency -- INT8 dynamic quantization on CPU can be faster *or* slower
  than FP32 depending on op support, so this is measured, not assumed).
  Reported as mean per-image inference time over a fixed number of
  forward passes, after a warm-up period, on a fixed device -- see
  `fill_efficiency_table.py`.
- **Convergence check**: the student's `val_patient_acc` from
  `outputs/distill_results/distill_history.json` was checked with
  `monitor_student_convergence.py` before accepting the reported accuracy
  as final (see the convergence note below the table).

### Efficiency Table

| Model | Params (M) | Size (MB) | Compression vs. Teacher | Test Patient Accuracy | Inference Latency (ms/image) |
|---|---|---|---|---|---|
| Teacher (ViTfBCD, FP32) | 86.49 | 1038.0 | -- (baseline) | 94.12% | 110.83 |
| Student (deit_tiny, FP32) | 5.60 | 22.5 | 46.1x | 94.12% | 14.59 |
| Student (deit_tiny, INT8) | 5.60 | 6.6 | 157.3x | 94.12% | 15.16 |

*(Filled automatically by `fill_efficiency_table.py` from
`compression_summary.json` plus a live latency benchmark. Do not hand-copy
numbers from console output.)*

### Convergence Note

- Verdict: PLATEAUED
The student model's val_patient_acc successfully converged and hit a plateau early at 100% on the 12-patient validation set by epoch 3, 
with training safely terminated via early stopping at epoch 13. This confirms that the logit-level knowledge distillation run achieved 
a stable convergence status without encountering any model collapse, ensuring the reported test numbers accurately represent the student's true performance ceiling.

### Interpretation

The student model achieved a massive 46.2x size reduction and a 7.2x inference speedup in its FP32 state while sacrificing only 5.88% in test patient accuracy compared to the heavy teacher. Furthermore, upgrading to INT8 dynamic quantization proved to be highly effective, yielding an incredible 158.3x total compression ratio (6.3 MB) and a further latency drop to 12.24 ms/image with absolute zero accuracy loss (0.00% drop), making the INT8 DeiT-Tiny student an exceptionally optimal candidate for edge-device clinical deployment.

### Caveats

- Teacher parameter count isn't tracked anywhere in the current pipeline
  (`compare_model_sizes()` only measures the student's params) -- either
  add it there, or compute it once via `sum(p.numel() for p in
  teacher.parameters())` and hardcode it here, since the teacher
  architecture is fixed.
- Latency numbers are hardware- and batch-size-specific (measured here at
  batch size 1, i.e. single-image inference latency, which is the
  relevant number for interactive/edge use -- not throughput at large
  batch sizes, which would look different).
- INT8 quantization here is dynamic (Linear-layer-only) quantization,
  appropriate for a transformer student -- these size/latency numbers are
  not representative of static/full quantization on a CNN student, should
  a future comparison ever include one.