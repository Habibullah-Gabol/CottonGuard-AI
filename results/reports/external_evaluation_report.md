# CottonGuard AI — External Evaluation

Report generated: 2026-10-29

## Experimental setup

- Model: fine-tuned ResNet18 checkpoint.
- Dataset: Cotton Leaf Disease Dataset with Severity Levels, original archive.
- Exact duplicate images excluded against the original dataset: 22.
- Eligible images: 958.
- Training on the external evaluation images: No.
- Preprocessing: resize to 224 × 224, tensor conversion, ImageNet normalization.

## Overall results

- Accuracy: 0.8800 (88.00%)
- Macro precision: 0.4199
- Macro recall: 0.4134
- Macro F1: 0.4158

## Class-wise results

| Class | Support | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| bacterial_blight | 119 | 0.7736 | 0.6891 | 0.7289 |
| curl_virus | 14 | 0.0000 | 0.0000 | 0.0000 |
| fussarium_wilt | 36 | 0.0000 | 0.0000 | 0.0000 |
| healthy | 789 | 0.9060 | 0.9645 | 0.9343 |

## Confusion matrix

Rows are true labels; columns are predicted labels.

| true_label       |   bacterial_blight |   curl_virus |   fussarium_wilt |   healthy |
|:-----------------|-------------------:|-------------:|-----------------:|----------:|
| bacterial_blight |                 82 |            2 |                0 |        35 |
| curl_virus       |                  0 |            0 |                0 |        14 |
| fussarium_wilt   |                  4 |            2 |                0 |        30 |
| healthy          |                 20 |            6 |                2 |       761 |

## Interpretation and limitations

The external evaluation shows a substantial difference between overall accuracy and macro F1. The healthy class is much larger than the disease classes, and the model fails to recognize the external curl-virus and Fusarium-wilt examples in this evaluation.

This result indicates limited cross-dataset generalization. It does not by itself establish why the errors occur. Exact SHA-256 duplicates were excluded, but resized or visually similar duplicates may remain. External dataset labels have not been independently verified.

The external evaluation must remain a fixed baseline. Do not use these images to select model settings and then report performance on the same images as an unbiased final test.

## Original model benchmark

Previously recorded original held-out test accuracy: 98.67%.
This value is reported separately and was not recalculated by this cell.