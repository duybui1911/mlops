from PIL import Image
from config import Config
from utils import get_index, search, display_html
from model import VIT_MSN
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
from google.cloud import storage
import uuid
import os
import uvicorn
import numpy as np

INDEX_NAME = Config.INDEX_NAME
index = get_index(INDEX_NAME)
logger.info(f"Connect to index {INDEX_NAME} successfully")

# Initialize GCS client
GCS_BUCKET_NAME = Config.GCS_BUCKET_NAME
storage_client = storage.Client()
try:
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    logger.info(f"Connected to GCS bucket {GCS_BUCKET_NAME} successfully")
except storage.exceptions.NotFound:
    logger.error(f"Bucket {GCS_BUCKET_NAME} not found in Google Cloud Storage.")
    raise HTTPException(status_code=404, detail=f"Bucket {GCS_BUCKET_NAME} not found.")
except Exception as e:
    logger.error(f"Error retrieving bucket {GCS_BUCKET_NAME}: {e}")
    raise HTTPException(status_code=500, detail=f"Error retrieving bucket {GCS_BUCKET_NAME}: {e}")

DEVICE = Config.DEVICE
model = VIT_MSN(device=DEVICE)
model.eval()
if DEVICE == "cuda":
    for param in model.parameters():
        param.data = param.data.float()
logger.info(f"Load model to {DEVICE} successfully")

app = FastAPI()

@app.post("/push_image/")
async def push_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # generate a unique id for the image
        unique_id = str(uuid.uuid4())
        file_extension = file.filename.split(".")[-1]
        gcs_file_path = f"images/{unique_id}.{file_extension}"

        # upload the file to GCS
        blob = bucket.blob(gcs_file_path)
        if not blob.exists():
            try:
                blob.upload_from_string(image_bytes, content_type=file.content_type)
                logger.info(f"Uploaded image to GCS successfully: {gcs_file_path}")
            except Exception as e:
                logger.error(f"Failed to upload image to GCS: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload image to GCS: {e}")
        else:
            logger.warning(f"Image already exists: {gcs_file_path}")

        feature = model.get_features([image]).reshape(1, -1)
        index.upsert([(
            unique_id,
            feature,
            {"gcs_path": gcs_file_path, "file_name": file.filename}
        )])
        logger.info(f"Upserted image to index successfully: {unique_id}")
        return {"message": "Successfully!", "file_id": unique_id, "gcs_file_path": gcs_file_path}
    except Exception as e:
        logger.error(f"Error in pushing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error in pushing image: {e}")

@app.get("/health_check/")
def health_check():
    return {"status": "OK!"}

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        raise HTTPException(status_code=400, detail=f"Error opening image: {e}")

@app.post("/find_image/")
async def find_image(file: UploadFile = File(...)):
    try:
        image = await upload_image(file)
        logger.info('Finding and displaying the similar images...')
        feature = model.get_features([image]).reshape(1, -1)
        match_ids = search(index, feature, top_k=Config.TOP_K)
        response = index.get_items(match_ids)
        images_url = [response[match_id]["metadata"]["url"] for match_id in match_ids]
        html_content = display_html(images_url)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error in finding image: {e}")
        raise HTTPException(status_code=500, detail=f"Error in finding image: {e}")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=False, workers=4)