from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.inference import run_inference

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG or PNG allowed")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file!")

    detections = run_inference(image_bytes)

    return JSONResponse(content={
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    })