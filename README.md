# RAG Pipeline for Paul Graham's Essays

A production-ready Retrieval-Augmented Generation (RAG) pipeline built from scratch. The project contains a robust custom HTML parser, vector embedding workflows utilizing localized deep-learning models, and a deterministic context retrieval search matrix paired with Groq API.

(All the terminal code lines written here are for bash use)

## System Architecture & Mechanics

[11 Online Essays] → (Custom HTML Scraper) → [Raw Text Files]
      ↓
(Text Chunking)
      ↓
[Local Database] ← (all-MiniLM-L6-v2) ← [118 Data Chunks]
      ↓
(User Query) → [Top-3 Similarity Check] → (Context Injection) → [Groq Llama 3.1 LLM]

### 1. Data Ingestion & Custom Processing Loop
* Target Source: 11 designated strategic essays scraped directly via raw HTTPS streaming protocols
* The HTML Parser Pipeline: Uses a modular `HTMLParser` overriding text nodes to aggressively filter layout frames, tables, and scripts, without relying on heavy third-party toolkits
* Text Slicing Blueprint: Elements are parsed into distinct windows of roughly 500 words each, featuring an explicitly declared 50-word slicing chunk overlap boundary to preserve text context across margins

### 2. The Vector Space Infrastructure
* The Embedding Compute Layer: Generates high-density geometric metrics locally via the `SentenceTransformer` frame running the `all-MiniLM-L6-v2` sequence layout topology
* The Vector Base: Each text block is mapped to a static 384-dimension matrix space. This layout runs operations strictly on localized CPUs, eliminating remote cloud processing
* Retrieval Math Engine: Incoming prompt parameters are converted using identical matrix layers. Chunks are evaluated using a vectorized `cosine_similarity` framework, returning the top `k=3` most similar content structures

### 3. LLM Orchestration & Context Injection Guardrails
* Inference Layer: Uses Groq Cloud Infrastructures running the `llama-3.1-8b-instant` execution frame
* Hallucination Countermeasures: System temperatures are hard-pinned to '0.01'. Moreover, the prompt templates use strict explicit markdown isolation frames, alongside a mandatory string boundary rule: `"If the answer is not in the context, say exactly: 'I don't have enough information to answer that.'"`

## Setting Up Local Runtime
- Install core vector compute dependencies, networking toolsets, and OpenAI compatibility SDKs
`pip install sentence-transformers scikit-learn requests python-dotenv openai numpy`
- Create a `.env` file within the base repository folder. Add your api key to: `GROQ_API_KEY=your_actual_private_groq_api_key_here`

## Running the Data Processing & Evaluation
Executes the scripts in the following chronological order to build and verify the data stores

### Step 1: Scrape and slice the Target Corpus - `python src/download_essays.py`
Downloads the latest version of the text streams, strips raw structural markups, splits data across boundaries, and creates `data/chunks.json`

### Step 2: Compute Vectors and Stand Up Database: `python src/embeddings.py`
Loads the 118 raw text snippets locally, passes them through the vector engine, and compiles the vector matrix registry inside `data/embeddings.json`

### Step 3: Run Interactive Search Terminals
Query the operational pipeline using the following command-line interface parameters:
    - Standard User Version: `python src/rag_chat.py --query` (In "" write any of the questions mentioned in the code regarding Paul Graham's essays)
    - Developer Diagnostics Mode - Prints the exact chunks from where the data to retrieve the information and their cosine scores: `python src/rag_chat.py --query "[Your Question]?" --verbose`
    - System Automated Test Suite - Executes inference routines across 10 required evaluation target queries and logs structural JSON performance data inside `data/evaluation_results.json`: `python src/rag_chat.py --evaluate`

## Deliverables Directory

The workspace structure contains the following critical components:
* `src/download_essays.py` - Fetches text data and handles initial chunk segmentation
* `src/embeddings.py` - Computes 384-dimension semantic vector models
* `src/retrieval.py` - Handles vector search and similarity calculations
* `src/rag_chat.py` - Connects the context database to the GROQ API model
* `data/evaluation_results.json` - Holds complete, unedited outputs across all evaluation runs
* `evaluation.md` - Contains manual failure-mode notes and parameter tuning insights