# CottonGuard AI 🌿

### Cotton Leaf Disease Classification with Deep Learning and Explainable AI

CottonGuard AI is a research prototype that classifies cotton leaf
images into four categories using a fine-tuned **ResNet18**
convolutional neural network. The repository also includes traditional
machine-learning baselines, dataset-audit and preparation utilities,
evaluation scripts, experiment notebooks, and Grad-CAM visual
explanations.

> **Important:** This is an experimental decision-support prototype, not
> a substitute for agricultural expertise, laboratory testing, or field
> diagnosis. Do not make pesticide or crop-treatment decisions based
> only on its predictions.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Application Screenshots](#application-screenshots)
-   [Supported Classes](#supported-classes)
-   [How It Works](#how-it-works)
-   [Repository Structure](#repository-structure)
-   [Results](#results)
-   [Installation](#installation)
-   [Run the Streamlit App](#run-the-streamlit-app)
-   [Grad-CAM Explainability](#grad-cam-explainability)
-   [Training and Evaluation](#training-and-evaluation)
-   [Data and Model Files](#data-and-model-files)
-   [Limitations](#limitations)
-   [Technology Stack](#technology-stack)
-   [Author](#author)

## Project Overview

Cotton leaf diseases can affect crop health and yield. CottonGuard AI
explores image-based classification as a way to help users review
possible leaf conditions.

The application provides: - Image upload and preview. - Four-class
prediction using a fine-tuned ResNet18 model. - A probability
distribution across the four classes. - Grad-CAM heatmap and overlay to
inspect image regions influencing a prediction. - A downloadable CSV
prediction report. - Traditional machine-learning baselines and
evaluation artifacts for research comparison.

## Application Screenshots

The following screenshots demonstrate the application's interface,
prediction workflow, and explainability feature.

### 1. Application Homepage

![CottonGuard AI Homepage](screenshots/01-homepage.png)

### 2. Image Upload and Preview

![Image Upload and Preview](screenshots/02-image-upload.png)

### 3. Disease Prediction Results

![Disease Prediction Results](screenshots/03-prediction-results.png)

### 4. Explainable AI --- Grad-CAM

![Grad-CAM Visualization](screenshots/04-gradcam-visualization.png)

### 5. Supported Disease Classes

![Supported Disease Classes](screenshots/05-disease-classes.png)

### 6. Model Architecture and Technology Stack

![Model Architecture and Technology
Stack](screenshots/06-model-architecture.png)

## Supported Classes

The class names and order below match the existing ResNet18 checkpoint.
The spelling `fussarium_wilt` is intentionally retained for
compatibility with the dataset labels and model.

  Model label          Display label
  -------------------- ------------------
  `bacterial_blight`   Bacterial Blight
  `curl_virus`         Curl Virus
  `fussarium_wilt`     Fusarium Wilt
  `healthy`            Healthy Leaf

## How It Works

1.  A user uploads a cotton leaf image through the Streamlit interface.
2.  The image is converted to RGB, resized to **224 × 224**, and
    normalized using ImageNet mean and standard deviation.
3.  The fine-tuned ResNet18 model produces class scores.
4.  The application displays the predicted class, estimated softmax
    confidence, and class probabilities.
5.  Grad-CAM can be generated to visualize image regions that
    contributed to the selected class score.

Grad-CAM is an interpretation of model behavior. It does not prove the
presence, location, or cause of a disease.

## Repository Structure

``` text
CottonGuard-AI/
├── app.py
├── app/
│   └── app.py
├── assets/
├── screenshots/
├── models/
│   ├── deep_learning/
│   │   └── resnet18_finetuned_best.pth
│   ├── traditional_ml/
│   ├── decision_tree_grouped.joblib
│   ├── knn_grouped.joblib
│   ├── random_forest_grouped.joblib
│   └── svm_grouped.joblib
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_traditional_ml.ipynb
│   ├── 03_resnet18_training.ipynb
│   ├── 04_model_explainability.ipynb
│   ├── 05_external_evaluation.ipynb
│   └── 06_results_verification.ipynb
├── results/
│   ├── audit_review/
│   ├── metrics/
│   ├── predictions/
│   └── reports/
├── src/
│   ├── config.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── train_resnet.py
│   ├── train_traditional_ml.py
│   ├── features/
│   └── evaluation/
├── tests/
├── requirements.txt
└── README.md
```

The repository excludes the original image datasets and the unused
`resnet18_best.pth` checkpoint. The fine-tuned inference checkpoint is
included.

## Results

The following figures are recorded in the project's saved evaluation
results. They describe the evaluated checkpoint and datasets; they
should not be interpreted as a guarantee of performance on new farms,
devices, or field conditions.

### Internal group-aware test

  Metric                     Result
  --------------------- -----------
  Test samples                  226
  Correct predictions     225 / 226
  Accuracy                   99.56%
  Macro F1-score             99.48%

### External Mendeley evaluation

  Metric              Result
  ----------------- --------
  Eligible images        958
  Accuracy            88.00%
  Macro F1-score      41.58%

The external evaluation is substantially weaker by macro F1, indicating
uneven performance across classes. The saved evaluation reports missed
external examples for the curl-virus and Fusarium-wilt classes. This
highlights the importance of testing across datasets and class
distributions rather than relying only on internal test performance.

Do not use the external evaluation set for model tuning and then report
it as an untouched test set.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Habibullah-Gabol/CottonGuard-AI.git
cd CottonGuard-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation may need to be matched to your CPU or CUDA
environment. If you already have a working PyTorch installation, avoid
reinstalling it unnecessarily.

## Run the Streamlit App

From the repository root, run:

``` bash
python -m streamlit run app/app.py
```

The app loads the existing checkpoint from:

``` text
models/deep_learning/resnet18_finetuned_best.pth
```

It performs inference; it does not train the model. If you use the
root-level `app.py` launcher instead, follow the entry-point behavior
defined in that file.

## Grad-CAM Explainability

The Streamlit application includes an optional **Explainable AI ---
Grad-CAM** section. After making a prediction: 1. Expand **Generate
visual explanation**. 2. Select **Generate Grad-CAM Heatmap**. 3. Review
the original image, heatmap, and overlay.

The visualization highlights regions that influence the selected class
score. Highlighted regions should be interpreted cautiously and are not
a disease segmentation mask.

## Training and Evaluation

Run commands from the repository root.

### Train traditional ML baselines

``` bash
python -m src.train_traditional_ml
```

### Evaluate saved traditional ML models

``` bash
python -m src.evaluation.evaluate_traditional_ml
```

### Evaluate the ResNet18 model

``` bash
python -m src.evaluation.evaluate_resnet
```

### Train / fine-tune ResNet18

``` bash
python -m src.train_resnet
```

**Caution:** The training script writes checkpoint files under
`models/deep_learning/`. Back up any checkpoint you want to preserve
before training.

### Generate a Grad-CAM visualization from the command line

``` bash
python -m src.evaluation.gradcam /path/to/image.jpg
```

### Run external evaluation

``` bash
python -m src.evaluation.evaluate_external
```

### Rebuild dataset splits

``` bash
python -m src.data.prepare_splits
```

**Caution:** Rebuilding splits regenerates dataset index and grouped
split artifacts. Run this only when you intentionally want to rebuild
them.

Some training and evaluation commands require dataset files that are not
included in this repository. See [Data and Model
Files](#data-and-model-files).

## Data and Model Files

-   The trained inference checkpoint is included at
    `models/deep_learning/resnet18_finetuned_best.pth`.
-   Traditional ML model artifacts are stored under `models/` and
    `models/traditional_ml/`.
-   The original image datasets are **not** included in this repository.
-   Some training/evaluation workflows expect grouped CSV files under
    `results/data/` and the original image dataset in a configured data
    directory.
-   The external Mendeley evaluation workflow expects the dataset at
    `external_dataset/Mendeley_Original/Cotton-Original/`, or at the
    location configured by the relevant script.

Check `src/config.py` and the individual script
arguments/environment-variable handling for the exact paths used by your
local setup before running data-dependent workflows.

## Limitations

-   Performance may vary with image quality, lighting, camera type, leaf
    age, cultivar, disease severity, and dataset distribution.
-   External macro F1 is notably lower than internal test macro F1, so
    class-wise generalization remains an important concern.
-   Softmax confidence is not a calibrated guarantee of correctness.
-   Grad-CAM provides a visual explanation of model activations, not
    causal evidence or a disease boundary.
-   Predictions should be reviewed by a qualified agricultural expert
    before any intervention.

## Technology Stack

-   **Python**
-   **PyTorch** and **Torchvision**
-   **ResNet18**
-   **Streamlit**
-   **Pandas** and **NumPy**
-   **Pillow**
-   **OpenCV** for image/Grad-CAM processing
-   **Scikit-learn** for traditional ML baselines and evaluation
-   **Jupyter Notebook** for experimentation

## Author

**Habibullah Gabol**

Computer Science Student \| Machine Learning & Deep Learning

-   **GitHub:** [Habibullah-Gabol](https://github.com/Habibullah-Gabol)
-   **Project Repository:** [CottonGuard
    AI](https://github.com/Habibullah-Gabol/CottonGuard-AI)

*Developed with a focus on machine learning, deep learning, and computer
vision.*
