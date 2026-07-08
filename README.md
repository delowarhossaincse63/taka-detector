# 🪙 Bangladeshi Taka (BDT) Detector API

Backend engine to automatically detect and classify Bangladeshi Taka banknotes from images using deep learning architectures.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00A4E4?style=for-the-badge&logo=analytics&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=Render&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)

---

### 🌐 Live Deployment
The API gateway is live, containerized, and fully production-tested:
👉 **Production Endpoint & Swagger Docs:** [https://taka-detector.onrender.com/docs](https://taka-detector.onrender.com/docs)

---

## 🚀 Key Features

* **Real-time Engine:** Leverages state-of-the-art computer vision to process and respond with high accuracy.
* **Denomination Supported:** Handles all core circulating BDT notes (10, 20, 50, 100, 200, 500, 1000 Taka).
* **Self-Documented Architecture:** Automated interactive user interfaces built natively with OpenAPI standards (`/docs` and `/redoc`).
* **Optimized Container:** Fully custom Docker image running Python-slim layers targeted for cloud memory bounds.
* **Security Resilience:** Hardened runtime layer bypassing modern PyTorch unpickling serialization boundaries.

---

## ⚙️ Architecture & Tech Stack

| Component | Framework / Library | Primary Purpose |
| :--- | :--- | :--- |
| **API Framework** | FastAPI + Uvicorn | High-performance request handling |
| **Deep Learning** | PyTorch + Ultralytics (YOLOv8) | Object detection & inference parsing |
| **Image Core** | Pillow (PIL) + OpenCV Headless | Memory buffer streaming & processing |
| **Environment** | Linux Alpine Core / Docker | Environment consistency and isolation |
| **Cloud Hosting**| Render Web Services | Serverlessly hosting backend clusters |

---

## 📂 Codebase Directory Layout

```text
taka-detector/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI main application & routes
│   └── inference.py       # YOLOv8 engine execution & safe global patches
├── model/
│   └── best.pt            # Pre-trained object detection weight binaries
├── Dockerfile             # Multi-stage production container instructions
├── requirements.txt       # Production pinned CPU-only package dependencies
└── README.md              # Documentation hub
🛠️ Step-by-Step Installation Guide
Follow these sequential steps to establish a local mirroring pipeline on your workstation.

Plaintext
[Step 1: Clone Repo] ──> [Step 2: Initialize Docker / Env] ──> [Step 3: Run Engine]
Approach A: Using Docker (Highly Recommended)
Docker isolates system architectures seamlessly without breaking global environments.

Bash
# 1. Clone the repository locally
git clone [https://github.com/delowarhossaincse63/taka-detector.git](https://github.com/delowarhossaincse63/taka-detector.git)
cd taka-detector

# 2. Build the optimized application image
docker build -t taka-detector .

# 3. Spin up the container instance mapped to standard web entry
docker run -p 7860:7860 taka-detector
Once initialized, hit the running local context at: http://localhost:7860/docs

Approach B: Standard Python Local Setup
Bash
# Initialize isolated python virtual instance
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependency tree targeted for CPU runtimes
pip install -r requirements.txt

# Fire up local development runtime cluster
uvicorn app.main:app --host 0.0.0.0 --port 7860
📡 Structured API Definition
1. Verification Ping
Method & Route: GET /

Response Signature:

JSON
{
  "message": "API is working!"
}
2. Run Note Inference
Method & Route: POST /predict

Content Payload: multipart/form-data

Form Argument Key: file (Supports raw .jpg, .jpeg, .png assets)

Streamlined Client Implementation (cURL)
Bash
curl -X 'POST' \
  '[https://taka-detector.onrender.com/predict](https://taka-detector.onrender.com/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_note.jpg;type=image/jpeg'
Production Payload Response Schema
JSON
{
  "filename": "sample_note.jpg",
  "total_detections": 1,
  "detections": [
    {
      "class_id": 4,
      "confidence": 0.9411,
      "bbox": {
        "x1": 7.99,
        "y1": 74.11,
        "x2": 495.11,
        "y2": 471.24
      }
    }
  ]
}
🤝 Contribution Guidelines
Found a bug or want to enhance the detection mapping accuracy? Open a Pull Request! We are currently working on hooking this robust pipeline directly into a dedicated mobile application (Flutter / React Native). Feel free to collaborate!

📄 License
Distributed under the permissive MIT License. See LICENSE for details.
