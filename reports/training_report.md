# Training Report: Task 2 (Transfer Learning)

Damaged is the positive class. Validation set: 56 images.

| Model | Accuracy | Precision | Recall | F1 | AP |
|---|---|---|---|---|---|
| Baseline CNN (threshold 0.5) | 0.696 | 0.704 | 0.679 | 0.691 | 0.787 |
| ResNet18 transfer (threshold 0.60) | 0.964 | 0.964 | 0.964 | 0.964 | 0.995 |

Pictures: assets/pr_curve.png, assets/confusion_matrix.png, assets/resnet_loss_curve.png

Notes:
- ResNet18 trained in 2 rounds: last layer only (5 epochs), then last block too (8 epochs).
- Augmentation: random crop, flip, rotation, colour change (train images only).
- Class weights used in the loss. Data is balanced now, but real club photos will not be.
- Threshold 0.60 chosen so recall stays at or above 90%, because missing damaged equipment is the costly mistake.
- Validation set is small, so scores can change by a few points between runs.
