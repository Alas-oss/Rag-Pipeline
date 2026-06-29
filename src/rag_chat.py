import os
import sys
import json
import argparse
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(__file__))
from retrieval import load_embeddings, retrieve

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Paul Graham's essays.
Answer the user's question using ONLY the context provided below.
Do not use any knowledge from outside the provided context.
If the answer is not in the context, say exactly:
"I don't have enough information to answer that."

CRITICAL: Be concise and factual. Provide complete sentences and stop when you have answered the question. Do not trail off or leave sentences incomplete."""




def build_prompt(query, chunks):
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Source {i+1} - {chunk['essay']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    return f"Context from Paul Graham's essays:\n\n{context}\n\nQuestion: {query}"


def generate_answer(query, context_chunks):
    user_message = build_prompt(query, context_chunks)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.01,
        max_tokens=600
    )
    finish_reason = response.choices[0].finish_reason
    content = response.choices[0].message.content

    if finish_reason == "length":
        print("Response was truncated because it reched max tokens.")
    return content


def rag_query(query, chunks, embeddings, model, k=3, verbose=False):
    relevant_chunks = retrieve(query, chunks, embeddings, model, k=k)

    if verbose:
        print("\n── Retrieved chunks ──────────────────────────────")
        for i, chunk in enumerate(relevant_chunks):
            print(f"[{i+1}] score={chunk['score']} | {chunk['essay']}")
            print(f"     {chunk['text'][:100]}...")
        print("──────────────────────────────────────────────────\n")

    answer = generate_answer(query, relevant_chunks)
    return answer, relevant_chunks


def run_evaluation(chunks, embeddings, model):
    test_questions = [
        "What does Paul Graham say about the importance of working on hard problems?",
        "How does Paul Graham define a startup?",
        "What advice does he give to young people choosing a career?",
        "What does he think about college education?",
        "How does Paul Graham view the role of investors?",
        "What does he say about procrastination?",
        "What is his view on programming languages?",
        "How does he describe the qualities of a good founder?",
        "What does he say about taste and aesthetics in work?",
        "What is his advice on how to get startup ideas?",
    ]
    results = []
    print("Running evaluation on 10 test questions...\n")

    for i, question in enumerate(test_questions):
        print(f"[{i+1}/10] {question[:60]}...")
        answer, retrieved = rag_query(question, chunks, embeddings, model, verbose=False)
        results.append({
            "question": question,
            "answer": answer,
            "sources": [{"essay": r["essay"], "score": r["score"]} for r in retrieved]
        })
        print(f"  Answer: {answer[:100]}...\n")

    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved evaluation results to data/evaluation_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG chatbot over Paul Graham essays")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--evaluate", action="store_true", help="Run all 10 test questions")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks")
    args = parser.parse_args()

    print("Loading model and embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunks, embeddings = load_embeddings()
    print(f"{len(chunks)} chunks loaded.\n")

    if args.evaluate:
        run_evaluation(chunks, embeddings, model)
    elif args.query:
        answer, retrieved = rag_query(args.query, chunks, embeddings, model, verbose=args.verbose)
        print(f"Answer:\n{answer}")
    else:
        print("Usage:")
        print("  python src/rag_chat.py --query \"your question here\"")
        print("  python src/rag_chat.py --query \"your question\" --verbose")
        print("  python src/rag_chat.py --evaluate")