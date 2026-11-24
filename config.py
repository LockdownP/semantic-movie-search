import os

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = "movies"

MODEL_NAME = "intfloat/multilingual-e5-small"

VECTOR_SIZE = 384

DATA_PATH = "data/movies.csv"

CLEAN_DATA_PATH = "data/movies_clean.csv"
