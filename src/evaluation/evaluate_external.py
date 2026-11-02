"""Evaluate the fixed checkpoint on Mendeley external images, excluding exact duplicates.

This script does not train or tune the model. Keep this external set separate
from model selection to preserve the meaning of the evaluation.
"""
import hashlib
import json
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from src.config import (
    CLASS_NAMES, ORIGINAL_DATA_DIR, EXTERNAL_DATA_DIR,
    PREDICTIONS_DIR, METRICS_DIR, REPORTS_DIR, IMAGE_EXTENSIONS,
    DEEP_LEARNING_MODELS_DIR, ensure_project_dirs,
)
from src.predict import get_device, get_transform, load_model


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_external_class(folder_name):
    key = folder_name.strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "bacterial blight": "bacterial_blight",
        "curl virus": "curl_virus",
        "fussarium wilt": "fussarium_wilt",
        "fusarium wilt": "fussarium_wilt",
        "healthy": "healthy",
    }
    return mapping.get(key)


def main():
    ensure_project_dirs()
    if not ORIGINAL_DATA_DIR.exists():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_DATA_DIR}")
    if not EXTERNAL_DATA_DIR.exists():
        raise FileNotFoundError(f"External dataset not found: {EXTERNAL_DATA_DIR}")

    original_hashes = set()
    for path in ORIGINAL_DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                original_hashes.add(sha256_file(path))
            except OSError:
                pass

    samples, duplicates, unreadable = [], 0, 0
    for path in sorted(EXTERNAL_DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(EXTERNAL_DATA_DIR)
        if len(relative.parts) < 2:
            continue
        true_label = normalize_external_class(relative.parts[0])
        if true_label not in CLASS_NAMES:
            continue
        try:
            if sha256_file(path) in original_hashes:
                duplicates += 1
                continue
            with Image.open(path) as image:
                image.verify()
            samples.append((path, relative, true_label))
        except Exception:
            unreadable += 1

    print("Exact duplicates excluded:", duplicates)
    print("Unreadable images skipped:", unreadable)
    print("Eligible external images:", len(samples))
    if not samples:
        raise ValueError(
            "No eligible images found. Check external folder names and mapping."
        )

    device = get_device()
    model = load_model(
        DEEP_LEARNING_MODELS_DIR / "resnet18_finetuned_best.pth", device
    )
    transform = get_transform()
    rows = []

    with torch.inference_mode():
        for index, (path, relative, true_label) in enumerate(samples, start=1):
            try:
                with Image.open(path) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
                predicted_index = int(probabilities.argmax())
                predicted_label = CLASS_NAMES[predicted_index]
                rows.append({
                    "image": path.name,
                    "relative_path": relative.as_posix(),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": float(probabilities[predicted_index]),
                    "correct": predicted_label == true_label,
                })
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
            if index % 100 == 0:
                print(f"Processed {index}/{len(samples)} images")

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("No images were successfully predicted.")

    y_true = predictions["true_label"]
    y_pred = predictions["predicted_label"]
    metrics = {
        "images_evaluated": int(len(predictions)),
        "correct_predictions": int(predictions["correct"].sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0
        )),
        "exact_duplicates_excluded": duplicates,
        "unreadable_images_skipped": unreadable,
    }

    predictions.to_csv(PREDICTIONS_DIR / "mendeley_external_predictions.csv", index=False)
    (METRICS_DIR / "mendeley_external_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        METRICS_DIR / "mendeley_external_confusion_matrix.csv"
    )
    report = classification_report(
        y_true, y_pred, labels=CLASS_NAMES, target_names=CLASS_NAMES,
        zero_division=0, output_dict=True,
    )
    (REPORTS_DIR / "mendeley_external_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nExternal evaluation (weights unchanged):")
    print(json.dumps(metrics, indent=2))
    print("\nClass-wise report:")
    print(classification_report(
        y_true, y_pred, labels=CLASS_NAMES, target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Saved predictions and metrics. Do not use this external set for tuning.")


if __name__ == "__main__":
    main()
