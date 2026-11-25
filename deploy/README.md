# Sports Equipment Condition Classifier - API

FastAPI service that classifies sports equipment photos as Functional or Damaged.
The PyTorch model was exported to ONNX and runs with onnxruntime. It is deployed on Vercel.

Endpoints:
- GET / : simple upload page
- POST /predict : send an image file, get JSON with label, confidence and needs_human_review
- GET /docs : interactive API docs

The web interface is hosted on a Hugging Face Static Space (Docker Spaces now need a paid plan).
See MODEL_CARD.md for performance, limitations and intended use.
convert_to_onnx.py shows how the model was exported.
