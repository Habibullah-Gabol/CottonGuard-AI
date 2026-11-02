"""Generate a Grad-CAM overlay for one image using the saved ResNet18 model."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.config import CLASS_NAMES, DEEP_LEARNING_MODELS_DIR
from src.predict import get_device, get_transform, load_model


def create_gradcam(image, model, device):
    """Return resized RGB image, normalized Grad-CAM, and class probabilities."""
    transform = get_transform()
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    activations, gradients = {}, {}

    def save_activation(_module, _inputs, output):
        activations["value"] = output

    def save_gradient(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = model.layer4.register_forward_hook(save_activation)
    backward_handle = model.layer4.register_full_backward_hook(save_gradient)

    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    target_index = int(logits.argmax(dim=1).item())
    logits[0, target_index].backward()

    feature_maps = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    forward_handle.remove()
    backward_handle.remove()

    return (
        np.asarray(image.convert("RGB").resize((224, 224))),
        cam,
        CLASS_NAMES[target_index],
        probabilities.detach().cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument(
        "--checkpoint",
        default=str(DEEP_LEARNING_MODELS_DIR / "resnet18_finetuned_best.pth"),
    )
    parser.add_argument("--output", default="gradcam_overlay.png")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    device = get_device()
    model = load_model(args.checkpoint, device)
    rgb, cam, predicted, probabilities = create_gradcam(image, model, device)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input image")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(rgb)
    axes[2].imshow(cam, cmap="jet", alpha=0.4)
    axes[2].set_title(f"Prediction: {predicted}")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160, bbox_inches="tight")
    print("Predicted class:", predicted)
    print("Probabilities:")
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"  {name}: {probability:.4f}")
    print("Saved overlay:", args.output)


if __name__ == "__main__":
    main()
