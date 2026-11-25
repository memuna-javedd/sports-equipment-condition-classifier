# Model Card: Sports Equipment Condition Classifier

## Model details
- Task: binary image classification (Functional vs Damaged)
- Architecture: small CNN, 3 conv blocks (16, 32, 64 filters) + dense layer, built in PyTorch, about 2.1M parameters
- Input: RGB image resized to 128x128
- Built for Falcons Sports Club as an intern project

## Intended use
Helps club staff decide which equipment to inspect first for maintenance. It is a screening aid only. A human must make the final decision.

## Out-of-scope use
- Safety-critical checks (helmets, protective gear, climbing or similar equipment)
- Automatically approving or rejecting equipment without human inspection
- Equipment types or conditions not represented in the training data
- Photos that do not show sports equipment

## Training data
- 276 images, balanced: 138 Functional and 138 Damaged
- Split: 220 training, 56 validation (80/20, seed 42)
- Functional images: Google Open Images V7 ("Sports equipment" class)
- Damaged images: collected from web image searches (e.g. broken bat, cracked helmet, torn net)
- Images from the web belong to their original owners and are used for educational purposes only

## Performance
- Validation accuracy: 69.6% (39 of 56 images)
- Training accuracy reached about 92%, so the model overfits
- The validation set is small, so results can vary by several points

## Limitations
- Very small dataset (276 images, fewer than the 500 planned)
- Moderate accuracy: it will make mistakes, including calling damaged items functional
- Damaged images from web searches may contain noise or duplicates
- Sensitive to lighting, camera angle and background
- May miss small damage such as hairline cracks or internal wear
- Does not recognise when a photo is not sports equipment
- Low-confidence predictions (below 70%) are flagged with needs_human_review = true

## Ethics and safety
- The most serious error is a false "functional" on damaged equipment, which could lead to injury. Always keep human inspection in the process.
- Uploaded images are processed in memory and not stored by this app.
- Do not upload photos containing people or personal information.
- Dataset images are third-party content used for education only.

## How to use
Open the web page and upload a photo, or call the API:

    curl -X POST -F "file=@photo.jpg" https://YOUR-SPACE-URL/predict

Example response:

    {"label": "damaged", "confidence": 0.82, "needs_human_review": false}