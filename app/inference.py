from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("model/best.pt")

def run_inference(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(source=image, conf=0.25, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_id": cls_id,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 2),
                    "y1": round(float(box.xyxy[0][1]), 2),
                    "x2": round(float(box.xyxy[0][2]), 2),
                    "y2": round(float(box.xyxy[0][3]), 2),
                }
            })
    return detections