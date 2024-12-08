import torch
class Config:
    INDEX_NAME = "mlops1-project"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOP_K = 5
    PORT_EXPOSE = 30000
    INPUT_RESLUTION = 384
    MODEL_PATH = "./models"
    GCS_BUCKET_NAME = "dynamic-branch-441814-f1-bucket"

    MAX_BATCH_SIZE_EMBEDDING = 32
    SERVER_SLEEP = 0.05
    # Config redis db
    REDIS_HOST = "0.0.0.0"
    REDIS_DB = 0
    REDIS_PORT = 6379
    REDIS_QUEUE = "extract_feature_queue"
    REQUEST_TIMEOUT = 0.5