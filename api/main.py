import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# ===== PASTE YOUR CNN CLASS FROM THE NOTEBOOK HERE (replace this one) =====
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
# ===========================================================================

CLASSES = ["damaged", "functional"]

model = BaselineCNN()
model.load_state_dict(torch.load("baseline_cnn_best.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

app = FastAPI(title="Sports Equipment Condition Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 5 MB).")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Please upload a valid image file.")
    with torch.no_grad():
        probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
    i = int(probs.argmax())
    conf = float(probs[i])
    return {
        "label": CLASSES[i],
        "confidence": round(conf, 4),
        "needs_human_review": conf < 0.7,
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Sports Equipment Condition Classifier</h2>
    <p>Upload a photo of sports equipment.</p>
    <input type="file" id="f" accept="image/*">
    <button onclick="go()">Classify</button>
    <pre id="out"></pre>
    <script>
    async function go(){
      const fd = new FormData();
      fd.append('file', document.getElementById('f').files[0]);
      const r = await fetch('/predict', {method:'POST', body:fd});
      document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
    }
    </script>"""