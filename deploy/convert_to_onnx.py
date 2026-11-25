# Run this in Google Colab. It turns baseline_cnn_best.pth into model.onnx
import torch
import torch.nn as nn
from google.colab import files

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

files.upload()
model = BaselineCNN()
model.load_state_dict(torch.load("baseline_cnn_best.pth", map_location="cpu"))
model.eval()
dummy = torch.randn(1, 3, 128, 128)
try:
    torch.onnx.export(model, dummy, "model.onnx", input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)
except TypeError:
    torch.onnx.export(model, dummy, "model.onnx", input_names=["input"], output_names=["output"], opset_version=17)
files.download("model.onnx")
