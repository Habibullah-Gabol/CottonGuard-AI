# CottonGuard AI — Refactored Project

CottonGuard AI classifies cotton leaf images into four labels using a fine-tuned ResNet18 model. The repository also contains traditional machine-learning baselines, dataset preparation utilities, and evaluation scripts.

## Classes

The exact class order used by the existing ResNet18 checkpoint is:

1. `bacterial_blight`
2. `curl_virus`
3. `fussarium_wilt`
4. `healthy`

The spelling `fussarium_wilt` is preserved to match the existing dataset labels and checkpoint.

## Project layout

```text
CottonGuard_AI_Refactored/
├── app.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── predict.py
│   ├── train_resnet.py
│   ├── train_traditional_ml.py
│   ├── data/
│   │   └── prepare_splits.py
│   ├── features/
│   │   └── image_features.py
│   └── evaluation/
│       ├── evaluate_resnet.py
│       └── evaluate_external.py
├── tests/
│   └── test_predict.py
├── notebooks/
│   └── CottonGuard_AI_Cleaned_Workflow.ipynb
└── tests/
```

The ZIP contains source code and documentation only. It does not contain your datasets or 44 MB model checkpoints. Copy these source files into your existing project, where `models/`, `results/`, and `external_dataset/` already exist.

## Required existing artifacts

For inference, place the trained checkpoint at:

`models/deep_learning/resnet18_finetuned_best.pth`

The following CSV files are used by training and evaluation scripts:

- `results/data/train_grouped.csv`
- `results/data/validation_grouped.csv`
- `results/data/test_grouped.csv`

The original image dataset is expected at a sibling folder named `cotton` next to the project directory (as in Google Drive), or set `COTTONGUARD_DATA_DIR` to the actual dataset location.

The external Mendeley dataset is expected at:

`external_dataset/Mendeley_Original/Cotton-Original/`

## Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch installation may need to be matched to your local CUDA/CPU environment. If you already have a working PyTorch installation, avoid reinstalling it without a reason.

## Run the app

From the project root:

```bash
streamlit run app.py
```

The app uses the existing checkpoint; it does not train the model.

## Useful commands

Run the internal grouped test evaluation:

```bash
python -m src.evaluation.evaluate_resnet
```

Train the traditional ML baselines:

```bash
python -m src.train_traditional_ml
```

Evaluate saved grouped traditional ML models:

```bash
python -m src.evaluation.evaluate_traditional_ml
```

Generate a Grad-CAM overlay for an image:

```bash
python -m src.evaluation.gradcam /path/to/image.jpg
```

Run the external evaluation:

```bash
python -m src.evaluation.evaluate_external
```

Train ResNet18 from scratch using the pretrained ImageNet initialization:

```bash
python -m src.train_resnet
```

**Warning:** `train_resnet.py` writes checkpoint files under `models/deep_learning/`. Back up existing checkpoints before running it if you want to preserve the current model.

Rebuild the deduplicated index and grouped splits:

```bash
python -m src.data.prepare_splits
```

**Warning:** This command regenerates `results/data/deduplicated_image_index.csv` and the three grouped split CSVs. Run it only if you intentionally want to rebuild them.

## Evaluation notes

Previously verified saved results for the existing checkpoint:

- Internal group-aware test: accuracy 99.56%, macro F1 99.48%, 225/226 correct.
- External Mendeley evaluation: accuracy 88.00%, macro F1 41.58% on 958 eligible images.

The external set is imbalanced and the checkpoint missed the external curl-virus and Fusarium-wilt examples in the saved evaluation. The app therefore labels its predictions experimental and warns users that softmax confidence is not a guarantee of correctness. Do not use the external evaluation set for model tuning and then report it as an untouched test.

## Important data caveat

The original notebook contains both an early random split and a later perceptual-group-aware split. This refactor treats the group-aware split CSVs as the intended evaluation artifacts. Avoid using the early `train.csv`, `validation.csv`, and `test.csv` when reporting leakage-resistant results.
