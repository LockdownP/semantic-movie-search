import time
import pandas as pd

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    MODEL_NAME,
    VECTOR_SIZE,
    CLEAN_DATA_PATH
)


def connect_qdrant():

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=10
    )

    # Wait until Qdrant is ready
    for attempt in range(60):

        try:
            client.get_collections()
            print("Qdrant is ready!")
            return client

        except Exception:
            print("Waiting for Qdrant...")
            time.sleep(2)

    raise RuntimeError("Cannot connect to Qdrant")


def ingest_data():

    # 1. Load dataset
    df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"Loaded {len(df)} movies")

    # 2. Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 3. Generate embeddings
    texts = [
        "passage: " + text
        for text in df["text"]
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding shape:", embeddings.shape)

    if embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError("Invalid vector dimension")

    # 4. Connect to Qdrant
    client = connect_qdrant()

    # 5. Create collection
    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print("Collection created!")

    # 6. Create payload indexes
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="genres",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="year",
        field_schema=models.PayloadSchemaType.INTEGER,
        wait=True
    )

    # 7. Create points
    points = []

    for i, row in enumerate(
        df.itertuples(index=False)
    ):

        point = models.PointStruct(
            id=int(row.id),
            vector=embeddings[i].tolist(),
            payload={
                "title": row.title,
                "year": int(row.year),
                "genre": row.genre,
                "genres": row.genre.split(),
                "overview": row.overview
            }
        )

        points.append(point)

    # 8. Insert points
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    # 9. Verify data
    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    print("Ingestion completed!")
    print("Total points:", count.count)


if __name__ == "__main__":
    ingest_data()
