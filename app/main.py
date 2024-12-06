from PIL import Image
from config import Config
from utils import get_index, search, display_html
from model import VIT_MSN
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
from google.cloud import storage
from google.oauth2 import service_account
import uuid
import time
import datetime

INDEX_NAME = Config.INDEX_NAME
index = get_index(INDEX_NAME)
logger.info(f"Connect to index {INDEX_NAME} successfully")

# Initialize GCS client
GCS_BUCKET_NAME = Config.GCS_BUCKET_NAME
# GOOGLE_SERVICE_ACCOUNT = os.getenv("GOOGLE_SERVICE_ACCOUNT")
# key_path = json.loads(GOOGLE_SERVICE_ACCOUNT)
# credentials = service_account.Credentials.from_service_account_info(key_path)
key_path = "dynamic-branch-441814-f1-45971c71ec3a.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
storage_client = storage.Client(credentials=credentials)
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
        # check 
        feature = model.get_features([image]).flatten().tolist() #reshape(1,-1)
        # match_ids = search(index, feature, top_k=Config.TOP_K)
        # if match_ids:
        #     match_vectors = []
        #     for match_id in match_ids:
        #         logger.info(f"Fetching match ID: {match_id}")
        #         match_item = index.fetch(ids=[match_id])
        #         if match_item and match_id in match_item:
        #             match_vectors.append(np.array(match_item[match_id]['values']))
        #         else:
        #             logger.warning(f"Match ID {match_id} not found in fetch response.")
        #     match_vectors_np = np.array(match_vectors)
        #     logger.info(f"Match vectors shape: {match_vectors_np.shape}")
        #     similarities = cosine_similarity([feature], match_vectors)
        #     for i, similarity in enumerate(similarities[0]):
        #         if similarity > Config.SIMILARITY_THRESHOLD:
        #             logger.info(f"Image already exists with ID: {match_ids[i]} and similarity: {similarity}")
        #             return {"message": "Image already exists!", "file_id": match_ids[i], "similarity": similarity}

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

@app.post("/image_search/")
async def image_search(file: UploadFile = File(...)):
    try:
        logger.info('Started image search process')
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        logger.info('Image successfully loaded and converted to RGB')

        feature = model.get_features([image]).flatten().tolist()
        start_time = time.time()
        match_ids = search(index, feature, top_k=Config.TOP_K)
        elapsed_time = time.time() - start_time
        logger.info(f'Search completed in {elapsed_time:.4f} seconds')

        response = index.fetch(ids=match_ids)
        # logger.info(f"Fetch response: {response}")
        images_url = []
        for match_id in match_ids:
            if match_id in response.get('vectors', {}):
                metadata = response['vectors'][match_id].get("metadata", {})
                gcs_path = metadata.get("gcs_path", "")
                blob = bucket.blob(gcs_path)
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(hours=1),
                    method="GET")
                images_url.append(signed_url)
                logger.info(f"Found URL for match ID {match_id}: {signed_url}")
            else:
                logger.warning(f"Match ID {match_id} not found in response.")
        return images_url
    except Exception as e:
        logger.error(f"Error in image search process: {e}")
        raise HTTPException(status_code=400, detail=f"Error in image search process: {e}")

@app.post("/display_image/")
async def display_image(file: UploadFile = File(...)):
    try:
        logger.info('Started image display process')
        images_url = await image_search(file)
        logger.info(f'Displaying {len(images_url)} similar images...')
        html_content = display_html(images_url)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error in displaying images: {e}")
        raise HTTPException(status_code=500, detail=f"Error in displaying images: {e}")