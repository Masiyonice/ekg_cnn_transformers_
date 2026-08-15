# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Binary classification of EKG images (Normal vs Abnormal, 580 images total) comparing a CNN transfer-learning
baseline against a hybrid CNN+Transformer model. All work — data loading, EDA, preprocessing, augmentation,
model definitions, training, and evaluation — lives in a single Jupyter notebook:
**`EKG_CNN_TRANSFORM.ipynb`**. There is no separate `src/` package; everything is notebook cells executed
top to bottom.

## Environment & commands

A local venv exists at `venv/` (Windows). Use its Python directly rather than a bare `python`:

```bash
"venv/Scripts/python.exe" -m pip list
```

Run/execute the notebook end-to-end in place (used after editing cells, e.g. for report regeneration):

```bash
"venv/Scripts/python.exe" -m jupyter nbconvert --to notebook --execute --inplace EKG_CNN_TRANSFORM.ipynb --ExecutePreprocessor.timeout=1800
```

Increase `--ExecutePreprocessor.timeout` if training more folds/epochs; a full 5-fold CV run for both models
takes a while even on GPU.

Key installed libs: `torch`/`torchvision` (CUDA build), `opencv-python`, `pandas`, `numpy`, `scikit-learn`,
`matplotlib`/`seaborn`, `python-docx` (for report generation). Check `torch.cuda.is_available()` in cell 1
output to confirm GPU is being used — CPU-only training is impractically slow for the Transformer model.

## Data layout

`dataset/` (git-ignored, must exist locally) contains four folders scanned by `FOLDER_MAP` in the notebook:

- `Abnormal Training`, `Abnormal Testing`, `Normal Training`, `Normal Testing`

**Important:** the original folder split is not used for train/test. All 580 images are pooled and
re-split via 5-fold Stratified Cross-Validation instead (the raw folders have a badly inverted Normal
split: 85 train / 291 test), so "Training"/"Testing" folder names are only a source label, not the actual
experimental split.

## Notebook architecture (cell groups, in execution order)

1. **Setup** — imports, seeds (`SEED=42`), `DEVICE` (cuda/cpu).
2. **Load Dataset** — `scan_dataset()` builds a DataFrame of `{filepath, label, split}`; `validate_images()`
   drops unreadable files.
3. **EDA** — class distribution, image dimensions, RGB/grayscale intensity checks (looking for color bias
   between classes).
4. **Preprocessing pipeline** (`preprocess_image()`) — fixed order: load RGB → grayscale → CLAHE
   (`clipLimit=2.0`, `tileGridSize=8x8`) → resize to 224×224 (`INTER_AREA`) → normalize to `[0,1]` float32.
   Individual steps (`load_image`, `to_grayscale`, `apply_clahe`, `resize_image`, `normalize_image`) are
   kept as separate functions for the step-by-step visualization cell.
5. **Augmentation** (`augment_image()` + helpers: rotate, shift, zoom, brightness, gaussian noise, blur) —
   class-specific intensity via `CLASS_AUG_MULTIPLIER` (`Abnormal` gets 1.5x the base params defined in
   `BASE_AUG_PARAMS`, since it's the minority class). Applied only to the train split, never test.
6. **PyTorch Dataset/DataLoader** — `EKGDataset` (general-purpose, single-channel) for pipeline sanity
   checks, and `EKGModelDataset` (used for actual training) which reads from a **precomputed cache**
   (`cached_images`, all 580 images preprocessed once into a numpy array) then augments on-the-fly per
   epoch and converts to 3-channel + ImageNet mean/std normalization for the pretrained backbone.
7. **Models**:
   - `build_cnn_baseline()` — `torchvision.models.resnet18` (ImageNet pretrained), all layers frozen except
     `layer4` and a replaced `fc` (2-class head).
   - `CNNTransformerHybrid` — same ResNet18 backbone with `avgpool`/`fc` stripped (outputs `(B,512,7,7)`),
     flattened into 49 tokens + a learned `[CLS]` token + learned positional embeddings, fed through a
     `nn.TransformerEncoder` (2 layers, 8 heads), then a linear classifier on the CLS output. Only
     `layer4` of the backbone is unfrozen (same as the baseline).
8. **Training** — `train_one_fold()` runs one model on one CV fold (Adam, fixed `lr=1e-4`, `EPOCHS=12`,
   `BATCH_SIZE_MODEL=16`, class-weighted `CrossEntropyLoss` to handle imbalance). `run_cross_validation()`
   (defined earlier in the CV section) drives `StratifiedKFold` over the cached images for a given
   model-builder function and returns per-fold metrics, histories, and concatenated true/pred/prob arrays.
9. **Evaluation & visualization** — metrics table (accuracy/precision/recall/f1/auc per fold, mean±std),
   confusion matrices, ROC curves, training curves, precision-recall-per-fold, calibration curve (Brier
   score), and Decision Curve Analysis (DCA) — all comparing CNN Baseline vs Hybrid side by side.

## Modeling decisions already made (do not silently change)

- **No fixed train/val/test split** — always 5-fold Stratified CV on the pooled 580 images. No separate
  validation set is held out within a fold.
- **No hyperparameter tuning** — epochs/LR/batch size are fixed and shared identically between both models
  and all folds, by explicit user choice.
- Both models use **transfer learning** (frozen ResNet18 backbone except `layer4`), not training from
  scratch — the dataset is too small for that.
- Class imbalance is handled via **loss class-weighting**, not oversampling. An earlier experiment with
  aggressive class-specific oversampling + heavy augmentation caused recall to collapse (train/test
  distribution shift from asymmetric augmentation) — avoid reintroducing that pattern; see the
  `Jhiro/dev-oversampling` branch history if revisiting synthetic-data approaches.
- If extending this project, hyperparameter tuning (LR, epoch count, unfreezing more layers, transformer
  dropout) is the natural next step, not architecture changes.

## Reports

`reports/` holds generated deliverables (docx/pdf write-ups, pseudocode, visualization docs) produced from
notebook outputs — these are build artifacts, not sources of truth. Regenerate from the notebook rather than
hand-editing them when results change.
