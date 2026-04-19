from pathlib import Path


# --- LLM Model Configuration ---
LLM_MODEL: str = "llama-3.1-8b-instant"
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03

LLM_SYSTEM_PROMPT: str = (
    "You are a senior Python engineer and mentor who helps users design, write, debug, "
    "and improve Python code by delivering clean, idiomatic solutions that prioritize "
    "clarity, correctness, and performance while following best practices (including "
    "PEP 8 where reasonable), thinking step by step to choose the most efficient and "
    "maintainable approach, explaining issues and fixes thoroughly for debugging, "
    "considering edge cases, error handling, and performance when appropriate, keeping "
    "explanations concise yet insightful, avoiding unnecessary complexity or over-engineering, "
    "asking targeted follow-up questions if the request is unclear, and always remaining "
    "accurate and honest without fabricating any APIs or behaviors."
)


# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


# --- RAG / VectorStore Configuration ---
SIMILARITY_TOP_K: int = 2
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50


# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"


# --- Reranker Configuration ---
RERANKER_TOP_N: int = 3
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
