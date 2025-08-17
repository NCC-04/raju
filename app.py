from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2, base64, os
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load YOLO model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
model = YOLO(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"image": "", "text": "❌ No image uploaded"}), 400

    try:
        # Read image
        file = request.files["image"]
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run YOLO prediction
        results = model(img)[0]

        # Collect detected labels
        labels = [results.names[int(cls)] for cls in results.boxes.cls]
        counts = Counter(labels)
        summary = ", ".join([f"{count} {label}{'s' if count > 1 else ''}" for label, count in counts.items()])
        if not summary:
            summary = "No objects detected"

        # Annotate image
        annotated = results.plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        b64_img = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"image": b64_img, "text": summary})

    except Exception as e:
        return jsonify({"image": "", "text": f"❌ Error: {str(e)}"}), 500

# Run Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
