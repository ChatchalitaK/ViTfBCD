## Cross-Dataset Generalization: BACH

### Motivation

All results reported so far are patient-wise splits *within* BreakHis. A
model can score well there while still relying on scanner-, staining-, or
site-specific artifacts that do not generalize to a different
histopathology dataset. To check this, we evaluate the primary model
**zero-shot** — frozen weights, inference only, no fine-tuning — on
[BACH/ICIAR2018](https://iciar2018-challenge.grand-challenge.org/), an
independent breast-tissue image dataset with its own scanners, staining
protocol, and patient population.

### Auxiliary Dataset Guidelines and Label Remapping

**Auxiliary datasets used, and their usage terms.**

| Dataset | Access | License / terms | Native format |
|---|---|---|---|
| BACH / ICIAR2018 | Requires registration at the [ICIAR2018 challenge site](https://iciar2018-challenge.grand-challenge.org/); downloaded here via `download_external_datasets.py` from the Zenodo mirror | CC BY-NC-ND — **non-commercial research use only**; no derivative redistribution; cite Aresta et al. (2019) | 400 `.tif` images, 4 classes (Normal, Benign, InSitu, Invasive), single fixed magnification |
| PCam (Camelyon16-derived) | Freely downloadable, no registration; fetched via `torchvision.datasets.PCAM` in `download_external_datasets.py` | Public / research-friendly; attribution to the original Camelyon16 challenge is good practice | 96×96 px H&E patches, binary label (tumor tissue present in the center 32×32 px region vs. not) |

Both datasets are used **strictly for zero-shot inference** (Section
"Setup") — neither contributes any training gradient to the primary model,
and neither is redistributed in processed form; only aggregate metrics and
confusion matrices are reported.

**Label remapping process.**

The primary model's binary head predicts {benign, malignant}. Neither
auxiliary dataset shares that exact label directly, so each is remapped
through a fixed, code-level mapping (not a judgment call made per-run):

*BACH* (`external_datasets.py: BACH_CLASS_TO_BINARY`):

| BACH class (folder name) | Remapped binary label |
|---|---|
| `Normal` | benign (0) |
| `Benign` | benign (0) |
| `InSitu` | malignant (1) |
| `Invasive` | malignant (1) |

*PCam* (`external_datasets.py: PCamBinaryDataset`): PCam's native label is
already binary (1 = tumor tissue present, 0 = no tumor) and is used directly
as malignant(1)/benign(0) with no remapping table — but see the semantic
caveat below.

*If evaluating an 8-class (subtype) checkpoint instead of the binary head*
(`cross_validation.py: SUBTYPE_TO_BINARY_IDX`, built from
`dataset.py: BENIGN_SUBTYPES`): the model's 8 subtype predictions are first
collapsed to binary before comparison against either auxiliary dataset —
`adenosis`, `fibroadenoma`, `phyllodes_tumor`, `tubular_adenoma` → benign;
the remaining four subtypes → malignant. This remap is applied to
*predictions only*; it does not change what the auxiliary dataset's ground
truth means.

**Semantic caveats on the remap itself** (methods-relevant, not just a
results caveat):

- BACH's Normal/Benign vs. InSitu/Invasive split is a *histologic diagnosis*
  boundary, distinct from BreakHis's benign/malignant *tumor subtype*
  boundary — the remap aligns them at the benign/malignant level only; it is
  not a claim that the two datasets' labeling protocols are equivalent.
- PCam's binary label is determined by tumor presence in a **32×32 px
  center crop of a 96×96 px patch**, not the whole patch. A PCam patch
  labeled "tumor" can therefore show mostly unremarkable tissue outside that
  center region — a coarser, spatially-different labeling rule than
  BreakHis/BACH's whole-image label. This is a likely contributor to any
  larger accuracy gap observed on PCam relative to BACH, independent of
  scanner or staining differences.
- No subtype-level (8-class) comparison is attempted against either
  auxiliary dataset, since neither provides subtype-equivalent ground truth.

### Setup

- **Model**: the primary ViTfBCD checkpoint from the standardized benchmark
  fold (see `benchmark_fold.json`), used strictly for inference — no weights
  are updated on BACH.
- **Label remap**: BACH's four classes (Normal, Benign, InSitu, Invasive) are
  collapsed to the same binary target used in training (Normal/Benign →
  benign, InSitu/Invasive → malignant).
- **Two conditions**, both using `cross_validation.py`:
  1. **No normalization** — raw BACH tiles, resized to the model's input size.
  2. **Macenko normalization** — BACH tiles normalized against the *same*
     fixed reference tile the primary model saw during training
     (`dataset.build_stain_normalizer`), isolating stain appearance as the
     variable under test.
- **Metrics**: accuracy, precision, recall/sensitivity, specificity, F1,
  AUC-ROC, and the confusion matrix, computed identically to the in-domain
  BreakHis evaluation.

### Results

> **FINALIZED** — this table is frozen (hash `4fdf7ab92a6309ff`, checkpoint `/home/user/Proj-Ploy/vit_breast_cancer/outputs/best_model.pt`, checkpoint file hash `dafcbb6de089d782`, frozen 2026-07-14T07:57:16.640510+00:00, git commit `unknown (not a git repo, or git unavailable)`). Re-running `cross_validation.py` will NOT change this table -- re-run `freeze_bach_results.py --force --reason "..."` explicitly if the underlying checkpoint or numbers genuinely change.

| Condition | n | AUC-ROC | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| No stain normalization | 400 | 56.8% | 55.4% | 69.0% | 44.5% | 0.5609 | 0.5813 |
| Macenko stain normalization | 400 | 58.0% | 61.6% | 42.5% | 73.5% | 0.5697 | 0.6051 |

**Recovery from normalization**: +1.25 percentage points of accuracy.

*(This table was filled and then frozen by `freeze_bach_results.py` -- see the FINALIZED banner above for the exact checkpoint and hash it corresponds to. It will not change on a routine `cross_validation.py` re-run.)*

### Interpretation

- If the no-normalization accuracy is substantially below in-domain BreakHis
  accuracy, that gap is evidence of a domain shift the model has not learned
  to ignore (scanner/stain appearance rather than tissue morphology).
- The benchmark-trigger convention used elsewhere in this project (see
  `cross_validation.py`) treats **recovering ≥5 points of accuracy**
  from normalization alone as a meaningful signal that stain appearance,
  specifically, was driving part of the gap — as opposed to a more
  fundamental generalization failure that normalization can't fix.
- `<fill: one or two sentences stating which of these patterns was observed,
  once the numbers above are in>`

### Caveats

- Licensing/usage terms and the exact label remap are documented in full
  under "Auxiliary Dataset Guidelines and Label Remapping" above; this
  section is for interpretation caveats only.
- BACH's Normal/Benign vs. InSitu/Invasive boundary is not the same
  clinical distinction as BreakHis's benign/malignant subtype boundary, so
  this is a generalization check, not a claim that the two datasets share
  ground-truth semantics.
- BACH has no ductal/lobular/mucinous/papillary subtype labels, so only the
  binary head is evaluated here — the 8-class subtype model is out of scope
  for this comparison.
- This section covers BACH only. PCam (a larger domain shift, per
  `cross_validation.py`) is a separate, not-yet-run comparison and
  should be added as its own subsection rather than folded into this one.
