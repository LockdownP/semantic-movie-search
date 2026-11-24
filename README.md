# Semantic Movie Search

A multilingual semantic movie search application.

## Tech Stack

- Python
- Qdrant
- Sentence Transformers
- Pandas
- Streamlit
- Docker Compose

## Features

- Thai and English semantic search
- Cosine similarity search
- Genre filtering
- Release year filtering
- Top-K results
- Similarity score threshold

## Getting Started

Requirements:
- Docker
- Docker Compose

Build and run:

    docker compose up --build -d

Open the application:

- Streamlit: http://localhost:8501
- Qdrant: http://localhost:6333/dashboard

Stop the application:

    docker compose down

## Embedding Model

intfloat/multilingual-e5-small

Vector size: 384 dimensions

## Dataset

15 movies with Thai descriptions.
