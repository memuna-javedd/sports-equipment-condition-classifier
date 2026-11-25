import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

CLASSES = ["damaged", "functional"]

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

app = FastAPI(title="Sports Equipment Condition Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def preprocess(img):
    img = img.resize((128, 128), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))[None, :, :, :]
    return x.astype(np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 4 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 4 MB).")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Please upload a valid image file.")
    logits = session.run(None, {input_name: preprocess(img)})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
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
