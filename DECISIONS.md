# Decision Log

## Why ResNet18 instead of the custom CNN
- The custom CNN overfit: about 92% train accuracy but only about 69.6% on validation. With 220 training images it cannot learn good features from scratch.
- ResNet18 already learned edges, textures and shapes from millions of ImageNet photos, so it only needs to learn the last step: functional or damaged.
- Result on the same validation images: F1 0.691 (baseline) vs 0.964 (ResNet18).
- ResNet18 is small and fast, which suits deployment on Hugging Face Spaces in Task 3.

## Other choices
- Precision-recall curve instead of accuracy: the club must not miss damaged equipment.
- Class-weighted loss: damaged items will be rarer in real use.
- Two-round training with a small learning rate for the pre-trained block, so it does not forget what it knew.
