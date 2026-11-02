"""Shared project paths and class configuration."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Override COTTONGUARD_DATA_DIR if the original dataset is stored elsewhere.
ORIGINAL_DATA_DIR = Path(
    os.environ.get("COTTONGUARD_DATA_DIR", PROJECT_ROOT.parent / "cotton")
)
EXTERNAL_DATA_DIR = PROJECT_ROOT / "external_dataset" / "Mendeley_Original" / "Cotton-Original"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"

DEEP_LEARNING_MODELS_DIR = PROJECT_ROOT / "models" / "deep_learning"
TRADITIONAL_MODELS_DIR = PROJECT_ROOT / "models" / "traditional_ml"

# Keep this order aligned with the original ResNet18 training/evaluation code.
CLASS_NAMES = [
    "bacterial_blight",
    "curl_virus",
    "fussarium_wilt",
    "healthy",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_project_dirs():
    """Create output directories, leaving datasets and model files untouched."""
    for directory in (
        RESULTS_DATA_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        DEEP_LEARNING_MODELS_DIR,
        TRADITIONAL_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
