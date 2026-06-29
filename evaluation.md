# Task 2 - RAG Pipeline Evaluation Report

## Pipeline Setup
* Chunking Strategy: about 500 words per chunk with a 50-word overlap
* Total Corpus Size: 11 Paul Graham essays were in total split into 188 chunks
* Embedding Model: 'all-MiniLM-L6-v2'
* Generation LLM: 'llama-3.1-8b-instant'
* Context Depth (k): Top-3 most similar chunks were injected per prompt

## Retrieval & Generation Accuracy
* Scode: 9/10 questions were successfully answered (90% accuracy)
* Groundedness: 100% grounded. At a temperature of 0.01, the model followed the system prompt properly and did not hallucinate.

## Analysis of Failure Cases

## Failure Case 1: Multi-Essay Information Scaterring (Question 3)
* Question: "What Advice does he give to your people choosing a career?"
* Result: The model answered using snippets from "Hiring is Obsolete", "What You'll Wish You'd Known", and "How to Do What You Love".
* Reason for Nuance: Since the answer was technically correct, there is no failure per se. But because the advice was scattered across multiple different essays, the setting for top-k results being 'k=3' limits how much total context the model can see. To get a deeped response, we can increase the 'k' to a higher value so that the model would pull more relevant chunks from the essays that we have.

## Failure Case 2: Missing Source Context (Question 9)
* Question: "What does he say about taste and aesthetics in work?"
* Result: "I don't have enough information to answer that."
* Reason for Failure: The source data was missing in the tracking pipeline and did not contain Paul Graham's specific essays on the topic. So the highest cosine similarity score retrieved was only '0.3559' for "How to do what you love". So since the information was actually missing, the LLM safe;y triggered the required fallback message instead of hallucinating.
