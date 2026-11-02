import unittest

import torch
from PIL import Image

from src.config import CLASS_NAMES
from src.predict import get_transform, predict_image


class DummyModel(torch.nn.Module):
    """A deterministic model used to test the prediction wrapper without weights."""
    def forward(self, batch):
        logits = torch.tensor([[0.1, 0.2, 0.3, 4.0]], device=batch.device)
        return logits.repeat(batch.shape[0], 1)


class PredictionTests(unittest.TestCase):
    def test_class_order_is_checkpoint_compatible(self):
        self.assertEqual(
            CLASS_NAMES,
            ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"],
        )

    def test_transform_returns_three_channel_224_image(self):
        image = Image.new("RGB", (300, 200), color=(100, 120, 80))
        tensor = get_transform()(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_predict_image_returns_expected_fields(self):
        image = Image.new("RGB", (300, 200), color=(100, 120, 80))
        result = predict_image(image, DummyModel(), device=torch.device("cpu"))
        self.assertEqual(result["predicted_class"], "healthy")
        self.assertIn("confidence", result)
        self.assertEqual(list(result["probabilities"]), CLASS_NAMES)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
