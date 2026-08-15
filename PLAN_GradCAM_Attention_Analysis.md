# Plan: Grad-CAM (CNN) vs Grad-CAM/Attention (Hybrid) Explainability Analysis + Report

## Goal

Add an explainability section to `EKG_CNN_TRANSFORM.ipynb` that shows *what each model looks at* when
classifying an EKG image, and uses it to give visual + quantitative evidence for the **local (CNN) vs
global (Transformer self-attention)** story behind the hybrid architecture:

- **CNN Baseline** → Grad-CAM heatmap on `layer4` (target = Abnormal logit).
- **CNN+Transformer Hybrid** → two complementary views:
  1. **Grad-CAM** on the shared ResNet18 backbone (`self.backbone[-1]` = `layer4`, target = Abnormal
     logit). Because gradients backprop *through* the Transformer to `layer4`, this is a **full-model**
     Grad-CAM anchored at `layer4` — not a backbone-only map. That is what makes CNN-vs-Hybrid Grad-CAM a
     fair same-layer / same-method comparison.
  2. **Transformer attention** (CLS → 49 patch tokens, rolled out across the 2 encoder layers) — the extra
     view unique to the hybrid, showing which spatial regions the global reasoning stage relied on.

All three maps live on the **same 7×7 spatial grid** (`(B,512,7,7)` backbone output → 49 tokens), so they
upsample to 224×224 identically and are spatially comparable cell-for-cell — a key strength to state
explicitly in the report.

This ties directly back to the existing 5-fold results (memory `modeling-decisions`: hybrid recall
0.637 vs 0.578, AUC 0.893 vs 0.870): the hypothesis is that the Transformer's global attention explains the
hybrid's better Abnormal detection.

## Constraints to respect (from existing project decisions)

- **No retraining required for the maps themselves.** Attention weights and Grad-CAM are functions of
  already-trained parameters; both can be produced by a modified *inference* forward pass. We do NOT change
  architecture or hyperparameters.
- Still 5-fold Stratified CV, no fixed val/test split — pin down exactly which fold's weights are used
  (Step 1) and state it, so the maps are reproducible and not overstated as ensemble-level.
- Keep everything in the notebook (no new `src/` package).
- Analysis-only: no hyperparameter tuning, no model changes.
- No new pip dependency — implement Grad-CAM and attention rollout by hand (the models are simple enough to
  hook directly; avoids `pytorch-grad-cam`).

## Steps

### 1. Fix which trained weights the maps use (reproducibility)
`run_cross_validation()` currently keeps only metrics/preds, not the fold model objects. Recommended,
clean and reproducible:
- Reuse the **same `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`** and retrain **one CNN
  and one Hybrid on fold 1's train split only**, with the identical fixed hyperparameters
  (`EPOCHS=12, lr=1e-4, batch=16`, class-weighted loss). Draw all visualized images from **fold 1's
  held-out split**, so "correct/misclassified" labels are honest (out-of-sample) rather than memorized.
- State clearly in the section + report: *maps come from fold-1 weights, illustrative not ensemble-averaged.*
- (Alternative, only if retraining time is a problem: modify `run_cross_validation()` to stash the fold-1
  models during the existing run. Retraining one fold per model is simpler and self-contained.)

### 2. Grad-CAM helper (shared by both models)
- Register a forward hook (save activations) and a full-backward hook (save gradients) on the target
  module: `model.layer4` for the CNN, `model.backbone[-1]` for the hybrid.
- Forward the image, take the **Abnormal logit (class index 1)**, `zero_grad`, `logit.backward()`.
- `weights = grad.mean(dim=(2,3))`; `cam = ReLU((weights[...,None,None] * activations).sum(1))`;
  upsample 7×7 → 224×224 (bilinear), min-max normalize to `[0,1]`, overlay on the preprocessed grayscale
  image.
- Implement as a small `GradCAM` class with an explicit `remove_hooks()` and call it in a
  `try/finally` (or context manager) so hooks never leak into later cells.

### 3. Transformer attention extraction (hybrid) — the tricky part
- **Gotcha:** `nn.TransformerEncoderLayer` internally calls `self_attn(..., need_weights=False)`, so the
  attention matrix is never computed and a plain forward hook returns `None`. You must actively recompute
  it. Robust options, in order of preference:
  1. Add a `forward_with_attention()` path that re-runs each encoder layer's `self_attn` manually with
     `need_weights=True, average_attn_weights=False` on the **same trained submodules** (reuse
     `layer.self_attn`, `layer.linear1/2`, norms, etc.), collecting per-layer `(heads, tokens, tokens)`
     weights. No retraining — same parameters, inference only.
  2. If reimplementing the layer is too fiddly, temporarily monkey-patch `self_attn.forward` to force
     `need_weights=True` and stash the returned weights via a hook, then restore. (More fragile — document
     the PyTorch version it was validated against: torch 2.12.)
- **Attention rollout** (Abnar & Zuidema): per layer average heads → `A`; add residual and renormalize
  `A_hat = normalize(0.5*A + 0.5*I)`; multiply across the 2 layers; take the **CLS row**, drop the CLS-self
  entry, giving a length-49 distribution over patches. Reshape 7×7, upsample, overlay (same format as
  Grad-CAM).
- Also compute the simpler **last-layer CLS attention (head-averaged)** as a sanity cross-check; report
  rollout as primary.
- Sanity asserts: attention rows sum ≈ 1; rollout is non-negative and sums ≈ 1 over the 49 patches.

### 4. Quantitative back-up for the local-vs-global claim (don't rely on eyeballing)
For each sample compute cheap scalar summaries of each 7×7 map and tabulate mean ± std across the sample set:
- **Concentration / spread**: normalized entropy of the 49-value map, or "effective area" (fraction of
  cells needed to reach 80% of total mass). Expectation: CNN Grad-CAM more concentrated (lower entropy),
  attention more distributed (higher entropy) — this is the numeric version of "patchy vs global."
- **Spatial agreement**: cosine similarity (or IoU of top-k cells) between the hybrid's Grad-CAM and its
  attention map, to show whether global attention attends beyond the locally salient region.
Keep it to a small table — this turns "analysis" from qualitative to defensible.

### 5. Sample-image selection (fixed, seeded, from fold-1 held-out set)
Pick 4–6 images covering:
- Normal correctly classified (both models),
- Abnormal correctly classified (both models),
- ≥1 **disagreement** case (ideally Abnormal that CNN misses but Hybrid catches — the recall-gap story),
  chosen from fold-1 held-out predictions if one exists; otherwise pick the lowest-confidence CNN Abnormal.

### 6. New notebook section "## 15. Explainability: Grad-CAM & Attention Analysis"
Insert after the DCA section (current last cell):
1. Markdown intro — purpose, methods, and the fold-1 caveat.
2. Fold-1 retrain of both models (Step 1).
3. `GradCAM` helper (Step 2).
4. Attention-rollout helper (Step 3).
5. Main figure: grid, rows = samples, cols =
   `[Original | CNN Grad-CAM | Hybrid Grad-CAM | Hybrid Attention Rollout]`.
6. Quantitative table (Step 4).
7. Data-driven interpretation markdown (mirroring the existing "Kesimpulan Otomatis" style): localized vs
   distributed maps, whether attention reaches beyond Grad-CAM's hotspot, and how that lines up with the
   hybrid's recall/AUC edge.

### 7. Re-execute the notebook
```bash
"venv/Scripts/python.exe" -m jupyter nbconvert --to notebook --execute --inplace EKG_CNN_TRANSFORM.ipynb --ExecutePreprocessor.timeout=1800
```
Raise the timeout if fold-1 retraining of both models pushes total runtime up.

### 8. Generate the report document (like before)
- Reuse the prior `python-docx` build-script pattern (`build_documentation.py`-style) used for
  `reports/*.docx`.
- Include: method write-up (Grad-CAM + attention rollout, the shared-7×7-grid fairness point, the fold-1
  caveat), the comparison figure(s), the quantitative table, and the interpretation.
- Export to PDF as done previously if that's the intended final format.

## Decisions to confirm before implementing
1. **Weights source:** retrain fold-1 models for the analysis (recommended) vs. stash models during the
   existing CV run?
2. **Report target:** new standalone report file vs. append a section to
   `reports/Laporan_Analisis_Klasifikasi_EKG_CNN_Transformer.docx`? ("like before" is ambiguous.)
3. **Attention mechanism:** OK to add a `forward_with_attention()` inference path to `CNNTransformerHybrid`
   (no retraining, no change to normal `forward()`), or prefer the monkey-patch approach?
4. **Samples:** fully seeded/random from fold-1 held-out, or any specific known cases you want included?
