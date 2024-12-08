import os
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
from config import Config
from dotenv import load_dotenv
import redis

load_dotenv()
PINECONE_APIKEY = os.getenv("PINECONE_APIKEY")


def get_index(index_name):
    pc = Pinecone(api_key=PINECONE_APIKEY)
    # if index_name in pc.list_indexes().names():
    #     pc.delete_index(index_name)
    #     logger.info(f"Deleted existing Pinecone index: {index_name}")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric="cosine",
            dimension=Config.INPUT_RESLUTION,
            spec=ServerlessSpec(
                cloud = "aws",
                region = "us-east-1"
            )
        )
        logger.info(f"Created Pinecone index: {index_name}")
    return pc.Index(index_name)

def search(index, input_emb, top_k):
    matching = index.query(vector=input_emb, top_k=top_k, include_values=True)[
        "matches"
    ]
    match_ids = [match_id["id"] for match_id in matching]
    return match_ids

def display_html(images_url):
    html_content = """
    <html>
        <head>
            <title>Images</title>
            <style>
                .image-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    grid-gap: 10px;
                }
                .image {
                    max-width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <div class="image-grid">
    """

    for url in images_url:
        html_content += f'<img src="{url}" alt="Image" width="200" height="300">'

    html_content += """
            </body>
        </html>
    """
    return html_content

def multi_pop(r, q, n):
    arr = []
    count = 0
    while True:
        try:
            p = r.pipeline()
            p.multi()
            for i in range(n):
                p.lpop(q)
            arr = p.execute()
            return arr
        except redis.ConnectionError as e:
            print(e)
            count += 1
            logger.error("Connection failed in %s times" % count)
            if count > 3:
                raise
            backoff = count * 5
            logger.info('Retrying in {} seconds'.format(backoff))
            time.sleep(backoff)
            r = redis.StrictRedis(
                host=cfgs.REDIS_HOST,
                port=cfgs.REDIS_PORT,
                db=cfgs.REDIS_DB
            )