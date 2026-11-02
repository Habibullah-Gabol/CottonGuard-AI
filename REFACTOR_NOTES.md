# Notebook cleanup notes

The uploaded `01_dataset_audit(1).ipynb` contains 123 cells. Many cells are repeated exploratory checks, outputs from earlier versions, or path-specific commands used while organizing the Drive project. The refactor moves repeatable operations into reusable Python modules and leaves exploration in the cleaned workflow notebook.

## Retained as reusable modules

- Dataset exact-duplicate audit and perceptual-hash group splitting: `src/data/prepare_splits.py`
- 72-dimensional Gabor + HSV feature extraction: `src/features/image_features.py`
- Grouped traditional ML training and validation comparison: `src/train_traditional_ml.py`
- Two-stage ResNet18 training (head training, then layer4 + head fine-tuning): `src/train_resnet.py`
- Internal grouped test evaluation: `src/evaluation/evaluate_resnet.py`
- External Mendeley evaluation with exact-duplicate exclusion: `src/evaluation/evaluate_external.py`
- ResNet18 inference and the user-facing Streamlit app: `src/predict.py`, `app.py`

## Consolidated or removed from the main workflow

- Repeated imports and repeated directory listings.
- Repeated checks of the same CSV columns and model checkpoint.
- Temporary one-off visualization cells for individual images.
- Duplicate implementations of traditional ML training and metrics.
- Early random train/validation/test splitting that was superseded by group-aware splitting.
- Drive-wide notebook searches and notebook-file inventory cells.
- Cells that installed packages in the middle of analysis; dependencies are listed in `requirements.txt`.
- Temporary code that launched Streamlit and tried Colab-specific proxy URLs. Launch the app with `streamlit run app.py` in a suitable terminal/runtime.

## Notes and safeguards

- The class spelling `fussarium_wilt` and class order are preserved for checkpoint compatibility.
- The source notebook's class order is explicitly documented in `src/config.py`.
- Training and split-generation scripts are opt-in because they can overwrite model or split artifacts.
- The uploaded notebook's external evaluation and internal test results are kept separate.
- The source notebook includes conflicting historical result values in some summary cells. This refactor documents the latest verified saved results from the conversation, but always recalculate from the actual checkpoint before publishing new results.
