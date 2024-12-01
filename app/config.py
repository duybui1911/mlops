import torch
class Config:
    INDEX_NAME = "mlops1-project"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOP_K = 5
    PORT_EXPOSE = 30000
    DATA_PATH = "./data"
    INPUT_RESLUTION = 384
    MODEL_PATH = "./models"
    GCS_BUCKET_NAME = "dynamic-branch-441814-f1-bucket"
    SIMILARITY_THRESHOLD = 0.95