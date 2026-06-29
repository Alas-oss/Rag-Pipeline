import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embeddings():
    with open("data/embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    embeddings = []

    for entry in data:
        chunks.append({
            "essay": entry["essay"],
            "text": entry["text"]
        })
        embeddings.append(entry["embedding"])

    embeddings = np.array(embeddings)
    return chunks, embeddings


def retrieve(query, chunks, embeddings, model, k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_k_indices:
        results.append({
            "essay": chunks[idx]["essay"],
            "text": chunks[idx]["text"],
            "score": round(float(scores[idx]), 4)
        })

    return results


if __name__ == "__main__":
    print("Loading model and embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    chunks, embeddings = load_embeddings()
    print(f"Loaded {len(chunks)} chunks\n")

    test_queries = [
        "What does Paul Graham say about working on hard problems?",
        "How does Paul Graham define a startup?",
        "What advice does he give to young people choosing a career?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        results = retrieve(query, chunks, embeddings, model, k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']} | {r['essay']}")
            print(f"       {r['text'][:120]}...")
        print()