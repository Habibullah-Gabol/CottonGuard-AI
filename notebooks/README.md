# CottonGuard-AI — Notebook Pack

This pack separates the original 137-cell notebook into topic-focused notebooks.

## Contents

1. `01_dataset_audit.ipynb` — cleaned dataset audit workflow: inventory, class distribution, image integrity, SHA-256 duplicates, cross-class duplicate review, pHash grouping, and candidate group-aware splits.
2. `02_traditional_ml.ipynb` — original traditional ML feature extraction and baseline experiments.
3. `03_resnet18_training.ipynb` — original ResNet18 training/fine-tuning workflow.
4. `04_model_explainability.ipynb` — Grad-CAM and background-sensitivity experiments.
5. `05_external_evaluation.ipynb` — external dataset evaluation and error analysis.
6. `06_results_verification.ipynb` — results verification and reporting utilities.

## How to use

1. Upload the `notebooks` folder to `My Drive/CottonGuard-AI/notebooks/`.
2. Open notebooks in numerical order in Google Colab.
3. Mount Google Drive where required.
4. Check all dataset, checkpoint, and results paths before executing.
5. Run one notebook at a time and inspect outputs before continuing.

## Important safety notes

- `01_dataset_audit.ipynb` writes new audit artifacts to `results/audit_review/`.
- It saves candidate splits separately and does **not** replace the existing production splits.
- Notebooks 02–06 retain experiment code from the supplied original notebook. They are reorganized, not independently revalidated end-to-end. Their original paths, variable dependencies, and artifact names must be checked before running.
- Training cells in notebook 03 may save or overwrite checkpoints. Back up existing model weights before running training.
- External evaluation in notebook 05 must remain an evaluation-only activity. Do not tune or select models using the external dataset.
- The ZIP contains notebooks and documentation only. It does not contain datasets, model checkpoints, or results.
