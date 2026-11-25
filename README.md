# sports-equipment-condition-classifier
Binary image classifier for sports equipment (Functional vs Damaged). Includes a curated 500-image dataset, OpenCV preprocessing script, and a baseline CNN built in PyTorch.

# Sports Equipment Condition Classifier

Binary image classifier that labels sports equipment photos as **Functional** or **Damaged**,
built for Falcons Sports Club. This repo covers **Task 1: Data Curation and Baseline Architecture**.

## Dataset
276 images, balanced: 138 functional and 138 damaged.

| Split | Functional | Damaged |
|-------|-----------|---------|
| train | 110 | 110 |
| val   | 28  | 28  |

Images are resized to 128x128 and normalized (pixels scaled to 0-1, then mean 0.5 / std 0.5).
The 80/20 split is random with a fixed seed (42).

### Data sources
- **Functional:** Google Open Images V7 ("Sports equipment" class), downloaded with FiftyOne.
  Kuznetsova et al., "The Open Images Dataset V4", IJCV 2020.
  https://storage.googleapis.com/openimages/web/index.html
- **Damaged:** [state honestly what you used: e.g. images collected from Google Image search using
  terms such as "broken cricket bat", "cracked helmet", "torn net"; plus any Roboflow or
  AI-generated images, with dataset names and licenses].
- Images from web searches belong to their original owners and are used here for educational purposes only.

## Project structure
```
data/        train/ and val/ folders, each with functional/ and damaged/
notebooks/   model.ipynb (preprocessing, CNN, training, loss curve)
models/      baseline_cnn.pth (final epoch), baseline_cnn_best.pth (epoch 6)
assets/      loss_curve.png
```

## Setup
```bash
git clone https://github.com/<your-username>/sports-equipment-condition-classifier.git
cd sports-equipment-condition-classifier
pip install -r requirements.txt
jupyter notebook notebooks/model.ipynb
```
The notebook was run on Google Colab (CPU). It expects the raw images in a zip named
`data-temp.zip` containing `functional/` and `damaged/` folders; the `data/` folder in this repo
is the processed output, so training can be re-run by pointing the loaders at `data/train`
and `data/val`.

## Baseline model
A simple CNN: 3 conv blocks (Conv, ReLU, MaxPool; 16, 32, 64 filters), then Flatten, Linear(128),
Dropout(0.5), Linear(2). About 2.1M parameters.
Training: 20 epochs, Adam (lr 1e-3), batch size 16, cross-entropy loss, random horizontal flip.

## Results
**Baseline validation accuracy: 69.6%** (39 of 56 val images).

![Loss curve](assets/loss_curve.png)

Training accuracy reaches about 92% while validation accuracy stays near 65-70% and validation
loss rises after epoch 6, so the model overfits. This is expected with 220 training images.
The val set is small (56 images), so scores vary by several points between runs.
This is the benchmark that Task 2 (transfer learning) will aim to improve.

## Limitations
- 276 images, fewer than the 500 in the project brief.
- The val set is small, so accuracy estimates are noisy.
- Some damaged images come from web searches and may include noise or duplicates.