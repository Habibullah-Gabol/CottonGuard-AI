"""Load the trained ResNet18 checkpoint and predict one cotton leaf image."""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.config import (
    CLASS_NAMES,
    DEEP_LEARNING_MODELS_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

DEFAULT_MODEL_PATH = DEEP_LEARNING_MODELS_DIR / "resnet18_finetuned_best.pth"


def get_device():
    """Use CUDA when available; otherwise run on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    """Match the preprocessing used in the original evaluation notebook."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(model_path=DEFAULT_MODEL_PATH, device=None):
    """Load a four-class ResNet18 state_dict without changing model weights."""
    device = device or get_device()
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Place resnet18_finetuned_best.pth in models/deep_learning/."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    # The supplied checkpoint is a plain state_dict, not a wrapped dictionary.
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_image(image, model, device=None):
    """Return predicted label, softmax confidence, and probabilities by class."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image object.")

    device = device or get_device()
    tensor = get_transform()(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }
