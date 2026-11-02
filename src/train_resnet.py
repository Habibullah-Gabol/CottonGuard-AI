"""Train/fine-tune ResNet18 using the existing group-aware CSV splits.

This script retrains and overwrites checkpoint filenames. Do not run it unless
you intentionally want to train again. Existing checkpoints are not needed to
run app.py; they are needed only for inference.
"""
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    CLASS_NAMES, RESULTS_DATA_DIR, METRICS_DIR,
    DEEP_LEARNING_MODELS_DIR, IMAGENET_MEAN, IMAGENET_STD,
    ensure_project_dirs,
)

SEED = 42
BATCH_SIZE = 32
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 8


class CottonDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform):
        self.frame = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.class_to_idx[row["label"]]


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, predictions, targets = 0.0, [], []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def train_stage(model, train_loader, val_loader, optimizer, criterion, device,
                epochs, checkpoint_path, stage_name):
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)

        print(
            f"{stage_name} {epoch + 1}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val accuracy={val_metrics['accuracy']:.4f} | "
            f"val macro-F1={val_metrics['macro_f1']:.4f}"
        )

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def main():
    ensure_project_dirs()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_df = pd.read_csv(RESULTS_DATA_DIR / "train_grouped.csv")
    val_df = pd.read_csv(RESULTS_DATA_DIR / "validation_grouped.csv")
    class_names = sorted(train_df["label"].unique())
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Training class order {class_names} differs from configured "
            f"checkpoint order {CLASS_NAMES}. Check labels before training."
        )
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CottonDataset(train_df, class_to_idx, train_transform)
    val_ds = CottonDataset(val_df, class_to_idx, eval_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Stage 1: train only the classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    model, history_head = train_stage(
        model, train_loader, val_loader, optimizer, criterion, device,
        HEAD_EPOCHS, DEEP_LEARNING_MODELS_DIR / "resnet18_best.pth",
        "Head training",
    )
    history_head.to_csv(METRICS_DIR / "resnet18_training_history.csv", index=False)

    # Stage 2: fine-tune layer4 and the classifier with smaller learning rates.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ])
    model, history_fine = train_stage(
        model, train_loader, val_loader, optimizer, criterion, device,
        FINETUNE_EPOCHS,
        DEEP_LEARNING_MODELS_DIR / "resnet18_finetuned_best.pth",
        "Fine-tuning",
    )
    history_fine.to_csv(METRICS_DIR / "resnet18_finetuning_history.csv", index=False)
    print("\nTraining complete. Checkpoints saved in models/deep_learning/.")


if __name__ == "__main__":
    main()
