"""Evaluate saved grouped traditional ML models on the grouped test split."""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    RESULTS_DATA_DIR, METRICS_DIR, TRADITIONAL_MODELS_DIR, ensure_project_dirs
)
from src.features.image_features import build_features


MODEL_FILES = [
    "random_forest_grouped.joblib",
    "decision_tree_grouped.joblib",
    "svm_grouped.joblib",
    "knn_grouped.joblib",
]


def main():
    ensure_project_dirs()
    test_frame = pd.read_csv(RESULTS_DATA_DIR / "test_grouped.csv")
    X_test, y_test = build_features(test_frame, "Grouped test")
    rows = []

    for filename in MODEL_FILES:
        model_path = TRADITIONAL_MODELS_DIR / filename
        if not model_path.exists():
            print(f"Skipping missing model: {model_path}")
            continue

        model = joblib.load(model_path)
        prediction = model.predict(X_test)
        rows.append({
            "Model": filename.removesuffix(".joblib"),
            "Accuracy": accuracy_score(y_test, prediction),
            "Macro Precision": precision_score(
                y_test, prediction, average="macro", zero_division=0
            ),
            "Macro Recall": recall_score(
                y_test, prediction, average="macro", zero_division=0
            ),
            "Macro F1": f1_score(
                y_test, prediction, average="macro", zero_division=0
            ),
            "Test Images": len(y_test),
        })

    if not rows:
        raise FileNotFoundError(
            "No grouped traditional ML model files were found in "
            f"{TRADITIONAL_MODELS_DIR}"
        )

    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    output = METRICS_DIR / "traditional_ml_grouped_test_results.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
