from functools import lru_cache

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    MODEL_NAME
)


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(url=QDRANT_URL)


def search_movies(
    query,
    top_k=5,
    genre=None,
    year_from=None,
    year_to=None,
    score_threshold=None
):

    if not query.strip():
        return []

    model = get_model()
    client = get_client()

    # 1. Convert query into vector
    query_vector = model.encode(
        "query: " + query.strip(),
        normalize_embeddings=True
    ).tolist()

    # 2. Build filters
    conditions = []

    if genre and genre != "All":

        conditions.append(
            models.FieldCondition(
                key="genres",
                match=models.MatchValue(
                    value=genre
                )
            )
        )

    if year_from is not None or year_to is not None:

        conditions.append(
            models.FieldCondition(
                key="year",
                range=models.Range(
                    gte=year_from,
                    lte=year_to
                )
            )
        )

    search_filter = (
        models.Filter(must=conditions)
        if conditions else None
    )

    # 3. Search Qdrant
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True
    )

    # 4. Return results
    return [
        {
            **point.payload,
            "id": point.id,
            "score": point.score
        }
        for point in response.points
    ]


if __name__ == "__main__":

    results = search_movies(
        query="หนังเกี่ยวกับอวกาศ",
        top_k=3
    )

    for movie in results:

        print(
            movie["title"],
            round(movie["score"], 4)
        )
