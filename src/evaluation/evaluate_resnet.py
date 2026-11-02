"""Evaluate the saved fine-tuned ResNet18 on the grouped internal test split."""
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from src.config import (
    CLASS_NAMES, RESULTS_DATA_DIR, METRICS_DIR, REPORTS_DIR,
    DEEP_LEARNING_MODELS_DIR, ensure_project_dirs,
)
from src.predict import load_model, get_device, get_transform


class TestDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = CLASS_NAMES.index(row["label"])
        return tensor, label


def main():
    ensure_project_dirs()
    csv_path = RESULTS_DATA_DIR / "test_grouped.csv"
    checkpoint_path = DEEP_LEARNING_MODELS_DIR / "resnet18_finetuned_best.pth"
    frame = pd.read_csv(csv_path)

    dataset = TestDataset(frame, get_transform())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    device = get_device()
    model = load_model(checkpoint_path, device)

    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Images": len(y_true),
        "Correct": sum(a == b for a, b in zip(y_true, y_pred)),
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / "resnet18_final_test_results.csv", index=False
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        METRICS_DIR / "resnet18_confusion_matrix.csv"
    )
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0, output_dict=True,
    )
    (REPORTS_DIR / "resnet18_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Device:", device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0,
    ))
    print("\nResults saved under results/metrics/ and results/reports/.")


if __name__ == "__main__":
    main()
