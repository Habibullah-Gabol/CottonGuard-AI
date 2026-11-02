# CottonGuard AI — Final Results Summary

**Report date:** 2026-10-29

## 1. Project overview

CottonGuard AI is a cotton leaf disease classification project using
a fine-tuned ResNet18 deep learning model and traditional machine-learning
baselines.

The four classes are:
- Bacterial blight
- Curl virus
- Fusarium wilt
- Healthy

## 2. Dataset preparation

- Original dataset images: 1,716
- Unique image contents after exact duplicate auditing: 1,591
- Training images: 1,126
- Validation images: 239
- Original held-out test images: 226
- External Mendeley images eligible after exact duplicate exclusion: 958

The original train, validation, and test split was group-aware to reduce
image leakage between splits.

## 3. Original held-out test results

The fine-tuned ResNet18 achieved:

- Accuracy: 98.67%
- Correct predictions: 223 of 226 images
- Macro precision: 98.85%
- Macro recall: 98.54%
- Macro F1-score: 98.69%

The traditional Random Forest baseline using 72 handcrafted features
achieved 96.02% accuracy and a macro F1-score of 95.28%.

## 4. External evaluation

The external dataset was evaluated without retraining or changing
the model weights.

- Number of images: 958
- Accuracy: 88.00%
- Macro precision: 41.99%
- Macro recall: 41.34%
- Macro F1-score: 41.58%

## 5. Interpretation

The external accuracy is substantially higher than the macro F1-score.
The dataset is imbalanced, with healthy leaves representing most examples.
The model missed all external Curl Virus and Fusarium Wilt examples.

Therefore, external accuracy alone does not adequately describe
performance across all disease classes. These results indicate a
generalization problem on this external dataset.

The evaluation does not establish the precise cause. Differences in image
appearance, dataset conditions, and label quality are possible factors
that would require further investigation.

## 6. Limitations and next steps

- The external evaluation is a separate dataset assessment, not a
  replacement for the original held-out test.
- The external dataset has already been evaluated and should not be used
  for model tuning while retaining its current status as an untouched test.
- Future improvements should be developed using separate training and
  validation data, then assessed on an independent test set.
- Results should be interpreted alongside per-class precision, recall,
  F1-score, and the confusion matrix.

## 7. Reproducibility

Key artifacts are stored in the project's results and models folders,
including the trained checkpoint, deduplicated image index, external
predictions, external metrics, confusion matrix, and external evaluation
report.
