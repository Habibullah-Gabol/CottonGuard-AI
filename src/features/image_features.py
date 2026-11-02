"""72-dimensional Gabor-texture and HSV-histogram features from the notebook."""
import cv2
import numpy as np


def extract_features(image_path, size=(128, 128)):
    """Extract 24 Gabor response statistics + 48 normalized HSV histogram values."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for sigma in [2.0, 4.0]:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F
            )
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.extend([
                float(response.mean()),
                float(response.std()),
                float(np.mean(np.abs(response))),
            ])

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    for channel, upper in zip(cv2.split(hsv), [180, 256, 256]):
        histogram = cv2.calcHist([channel], [0], None, [16], [0, upper]).flatten()
        histogram /= histogram.sum() + 1e-8
        features.extend(histogram.tolist())

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (72,):
        raise RuntimeError(f"Expected 72 features, got {feature_array.shape}")
    return feature_array


def build_features(frame, description="Dataset"):
    """Convert a split DataFrame with path/label columns into X and y arrays."""
    features, labels = [], []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        features.append(extract_features(row.path))
        labels.append(row.label)
        if index % 100 == 0 or index == len(frame):
            print(f"{description}: {index}/{len(frame)} images")
    return np.asarray(features, dtype=np.float32), np.asarray(labels)
