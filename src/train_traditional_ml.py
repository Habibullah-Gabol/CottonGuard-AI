"""Train traditional ML baselines on grouped training data and select by validation macro-F1."""
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    RESULTS_DATA_DIR, METRICS_DIR, TRADITIONAL_MODELS_DIR, ensure_project_dirs
)
from src.features.image_features import build_features


def main():
    ensure_project_dirs()
    train_df = pd.read_csv(RESULTS_DATA_DIR / "train_grouped.csv")
    val_df = pd.read_csv(RESULTS_DATA_DIR / "validation_grouped.csv")

    X_train, y_train = build_features(train_df, "Training")
    X_val, y_val = build_features(val_df, "Validation")

    candidates = {
        "knn_grouped": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "decision_tree_grouped": DecisionTreeClassifier(random_state=42),
        "svm_grouped": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1, probability=True, random_state=42)),
        ]),
        "random_forest_grouped": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }

    rows = []
    for name, model in candidates.items():
        started = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro Precision": precision_score(
                y_val, predictions, average="macro", zero_division=0
            ),
            "Macro Recall": recall_score(
                y_val, predictions, average="macro", zero_division=0
            ),
            "Macro F1": f1_score(
                y_val, predictions, average="macro", zero_division=0
            ),
            "Training Time (seconds)": round(time.time() - started, 2),
        })
        joblib.dump(model, TRADITIONAL_MODELS_DIR / f"{name}.joblib")

    results = pd.DataFrame(rows).sort_values(
        "Macro F1", ascending=False
    ).reset_index(drop=True)
    results.to_csv(METRICS_DIR / "grouped_validation_results.csv", index=False)
    print(results.to_string(index=False))
    print("\nBest validation model:", results.iloc[0]["Model"])
    print("Models and validation results saved.")


if __name__ == "__main__":
    main()
