from fastapi import FastAPI, File, UploadFile, HTTPException, Header, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, JSONResponse
from paint_converter import process_image
import cv2
import numpy as np
import io
import aiohttp

API_KEY = "your-secret-api-key"  # Replace with secure env var in prod
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/process-section/")
async def process_section(
        file: UploadFile = File(...),
        id: str = Form(...),
        x_api_key: str = Header(...)
):
    verify_api_key(x_api_key)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Process the image and perform callback inside process_image
    result, status = await process_image(image, id)

    if result is None:
        raise HTTPException(status_code=500, detail=f"Processing failed: {status}")

    return JSONResponse(status_code=200, content={"status": "processing complete", "id": id})
