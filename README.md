# RAG-Pro-Engine-Chatbot


Meet a production-ready RAG chatbot that truly knows Python  a helpful expert you can rely on. It’s built with LlamaIndex and Groq (running Llama 3.3, Llama 3.1, and Gemma 2), uses Hugging Face embeddings, and ships in a Streamlit app. Its answers are tested with Ragas for quality and reliability. Ask it anything about Python and if you probe it with something outside its sources, it will politely decline rather than guessing.

---
## 📁 Project structure
```
AI-ENGINEERING/
├── data/                          # Document storage
│   └── (Data-Quality-Fundamentals,Python notes,Python and Tkinter Programming.PDF)
├── evaluation/                    # Evaluation components
│   ├── __pycache__/
│   ├── evaluation_embedding_models/
│   ├── evaluation_results/
│   ├── evaluation_vector_stores/
│   ├── __init__.py
│   ├── evaluation_config.py
│   ├── evaluation_engine.py
│   ├── evaluation_helper_functions.py
│   ├── evaluation_model_loader.py
│   └── evaluation_questions.py
├── local_storage/                 # Local storage for embeddings and models
│   ├── embedding_model/
│   ├── vector_store/
│   └── notebooks/
└── src/                          # Source code
    ├── __pycache__/
    ├── __init__.py
    ├── config.py                  # Configuration settings
    ├── engine.py                  # RAG engine logic
    ├── model_loader.py            # Model initialization
    ├── .env                      # Environment variables
    ├── .gitignore
    ├── app.py                    # Main Streamlit application
    ├── environment.yml           # Conda environment
    ├── evaluate.py               # Evaluation scripts
    └── main.py                  # Main entry point
```
---
## 🏛️ Architecture

```
┌──────────────────────┐
│  User question       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     (optional)
│  HyDE query rewrite  │ ─── generates a hypothetical answer
└──────────┬───────────┘     and embeds it as the query
           │
           ▼
┌──────────────────────┐
│  Vector retriever    │     HuggingFace all-MiniLM-L6-v2 embeddings
│  (top-k passages)    │     over LlamaIndex's default vector store
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     (optional)
│  Cross-encoder       │ ─── ms-marco-MiniLM-L-6-v2 rerank
│  reranker            │     keeps only the top-N most relevant
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Groq LLM            │     System prompt (persona) +
│  (llama-3.3-70b)     │     retrieved context + chat memory
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Answer + sources    │
└──────────────────────┘
```
---
## 𑿯 Brisk start
### 1. Installation and environment setup
```bash
cd 
conda env create -f environment.yml
conda activate rag-project-env
```
---
### 2. Configure API key

Copy `.env.example` to `.env` and paste in a Groq API key from [console.groq.com](https://console.groq.com):

```bash
cp .env.example .env
# then edit .env
```
### 3. Use the CLI to run the chatbot (optional)

```bash
streamlit run app.py
```
### 4. Run the chatbot using Streamlit

```bash
python main.py
```
---
The **Ragas evaluation harness** runs your chatbot against a set of ground-truth questions defined in [`evaluation/evaluation_questions.py`](evaluation/evaluation_questions.py) and automatically scores each answer using four key metrics:

| Metric                | What it measures |
|-----------------------|------------------|
| **Faithfulness**      | How well the answer stays grounded in the retrieved context (no hallucinations) |
| **Answer Correctness**| How semantically close the answer is to the ground-truth answer |
| **Context Precision** | How relevant/on-topic the retrieved context is |
| **Context Recall**    | How much of the ground-truth information is covered by the retrieved context |

### How to run it

Simply execute:

```bash
python evaluate.py
```
Results
The evaluation results are saved in the `evaluation/evaluation_results/` folder as timestamped CSV files:

One detailed file with per-question results
One summary file with averaged scores for the entire experiment

> Note: By default, the harness includes rate limiting because Groq’s free tier is quite strict. If you're using a paid Groq tier or running with a local model, you can disable rate limiting by switching to the `evaluate_without_rate_limit` function in `evaluation/evaluation_engine.py./`
---
## 🛠️ Configuration guide
Default settings are in [src/config.py](src/config.py). The Streamlit sidebar exposes the most useful options at runtime.

| Setting                   | Default                                    | Function|
|---------------------------|--------------------------------------------|---------|
| `LLM_MODEL`               | `llama-3.3-70b-versatile`                  | Default Groq model. |
| `AVAILABLE_LLM_MODELS`    | Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B     | Models shown in the UI selector. |
| `LLM_TEMPERATURE`         | `0.1`                                      | Lower = more deterministic answers. |
| `EMBEDDING_MODEL_NAME`    | `sentence-transformers/all-MiniLM-L6-v2`   | Fast, high-quality sentence encoder. |
| `SIMILARITY_TOP_K`        | `4`                                        | Passages retrieved per question. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50`                          | Sentence-splitter configuration. |
| `RERANKER_MODEL_NAME`     | `cross-encoder/ms-marco-MiniLM-L-6-v2`     | Cross-encoder used when reranking is enabled. |
| `RERANKER_TOP_N`          | `3`                                        | Passages kept after the reranker. |
| `USE_HYDE_DEFAULT`        | `False`                                    | Start with HyDE off. |
| `CHAT_MEMORY_TOKEN_LIMIT` | `3900`                                     | Rolling chat memory window. |

---
## ⛯ Plan — next steps

This repo is a starting point. Two clear directions from the "What's Next?" brief:

**Path 1 — AI Engineer (deeper optimisation)**

- Evaluate alternative Groq / OpenAI / local LLMs, measuring Faithfulness vs. Answer Correctness.  
- Swap embedding models via the [Hugging Face MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and re-run evaluations.  
- Compare the built-in `HyDEQueryTransform` with the custom rewriter in [src/query_rewrite.py](src/query_rewrite.py).  
- Implement an LLM-based reranker (have the LLM score passages 1–10) and benchmark it against the cross-encoder.

**Path 2 — Product Builder (already delivered here)**

- Streamlit chat UI with sources, "New Chat", thumbs-up/down, and a full configuration sidebar.  
- Next: connect feedback buttons to an analytics backend, add authentication, and deploy to Streamlit Community Cloud.
---


