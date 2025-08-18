import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify,render_template
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load YOLO model
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image part"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Convert to OpenCV image
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # YOLO detection
        results = model(img)
        result = results[0]  # first image
        boxes = result.boxes  # Boxes object
        detections = []

        # Collect class names for detected objects
        class_names = [result.names[int(boxes.cls[i])] for i in range(len(boxes))]
        from collections import Counter
        counts = Counter(class_names)
        text_summary = ", ".join(f"{v} {k}s" if v > 1 else f"{v} {k}" for k, v in counts.items()) or "No objects detected"

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            name = result.names[cls_id]
            detections.append({
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": conf,
                "class": name
            })
            # Draw boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        text = ", ".join([det['class'] for det in detections]) or "No objects detected"

        return jsonify({"image": img_base64, "text": text_summary})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
 port = int(os.environ.get("PORT", 10000))
 app.run(host="0.0.0.0", port=port)  

