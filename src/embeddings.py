import json
import os
import numpy as np 
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_chunks():
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
def embed_chunks(chunks):
    print(f"LOading mode: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks")
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"Done. Each embedding has {embeddings.shape[1]} dimensions.")
    return model, embeddings

def save_embeddings(chunk, embeddings):
    data = [] 
    for i, chunk in enumerate(chunks):
        data.append({
            "essay": chunk["essay"],
            "text": chunk["text"],
            "embedding": embeddings[i].tolist()
        })

    with open("data/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} embeddings to data/embeddings.json")

if __name__=="__main__":
    chunks = load_chunks()
    model, embeddings = embed_chunks(chunks)
    save_embeddings(chunks, embeddings)