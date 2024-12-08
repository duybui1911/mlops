# Batch processing
import os
import redis
import json
import yaml
import time
import configparser
import base64
import cv2
import torch
import numpy as np

from PIL import Image
from loguru import logger
import utils
from config import Config
from model import VIT_MSN

db = redis.StrictRedis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB
)


class FeatureExtraction():
    def __init__(self, ):
        DEVICE = Config.DEVICE
        self.extractor = VIT_MSN(device=DEVICE)
        self.extractor.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        while True:
            t0 = time.time()
            # Get batch requests
            raw_requests = utils.multi_pop(
                db, Config.REDIS_QUEUE, Config.MAX_BATCH_SIZE_EMBEDDING)

            verified_request = []
            # step 1: parsing requests
            for req in raw_requests:
                if req is None:
                    continue
                req_payload = json.loads(req)
                verified_request.append(req_payload)

            if len(verified_request) == 0:
                # print("Sleep")
                time.sleep(Config.SERVER_SLEEP)
                continue

            logger.info("Loaded success")

            # Decode images from base64
            images = []
            image_ids = []
            for payload in verified_request:
                img_str = payload.image
                img_shape = (payload.height, payload.width)
                img_arr = np.frombuffer(
                    base64.decodebytes(img_str),
                    dtype=np.uint8
                )
                img_arr = img_arr.reshape(img_shape)
                images.append(Image.fromarray(img_arr).convert("RGB"))
                image_ids.append(payload.id)

            start_extract = time.time()
            features = self.extractor.get_features(images)
            end_extract = time.time()
            
            for (img_id, feat) in zip(image_ids, features):
                db.set(img_id, base64.b64encode(feat.astype(np.float32)).decode("utf-8"))

            logger.info("Time extract feature: {}".format(end_extract - start_extract))
            logger.info("Time Extract: {}".format(time.time() - t0))
            
            verified_request.clear()

            # sleeppppp...
            time.sleep(Config.SERVER_SLEEP)

if __name__ == "__main__":
    runner = FeatureExtraction()
    runner.run()