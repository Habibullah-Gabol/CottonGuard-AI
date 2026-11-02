
"""
Grad-CAM explainability for CottonGuard AI.

Uses the trained ResNet18 layer4 activations to generate
a heatmap showing image regions influencing a prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from PIL import Image

from src.predict import get_transform, get_device


def generate_gradcam(image, model, class_idx=None, device=None):
    """
    Generate Grad-CAM heatmap and overlay for a PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
        Original cotton leaf image.
    model : torch.nn.Module
        Loaded, trained ResNet18 model.
    class_idx : int, optional
        Target class index. Defaults to predicted class.
    device : torch.device, optional
        CPU or CUDA device.

    Returns
    -------
    dict
        Contains predicted class index, target class index,
        heatmap, and overlay as RGB NumPy arrays.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")

    device = device or get_device()

    image = image.convert("RGB")
    input_tensor = (
        get_transform()(image)
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    activations = {}

    # Capture activations from ResNet18 layer4.
    def forward_hook(module, inputs, output):
        activations["features"] = output

    handle = model.layer4.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        # Gradients must remain enabled for Grad-CAM.
        with torch.enable_grad():

            input_tensor.requires_grad_(True)

            logits = model(input_tensor)

            predicted_idx = int(
                logits.argmax(dim=1).item()
            )

            if class_idx is None:
                class_idx = predicted_idx

            score = logits[:, class_idx].sum()

            feature_maps = activations["features"]

            # Calculate gradients of target score
            # with respect to layer4 feature maps.
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=feature_maps,
                retain_graph=False,
                create_graph=False
            )[0]

            # Global average pooling of gradients.
            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            # Weighted combination of feature maps.
            cam = (weights * feature_maps).sum(dim=1)

            cam = F.relu(cam)

            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            cam = cam[0, 0].detach().cpu().numpy()

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    # Normalize heatmap.
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Resize original image for visualization.
    original = np.asarray(
        image.resize((224, 224))
    )

    # Combine original image and heatmap.
    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return {
        "predicted_idx": predicted_idx,
        "target_idx": class_idx,
        "heatmap": heatmap,
        "overlay": overlay,
    }
