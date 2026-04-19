import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

from src.config import (
    LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    CHAT_MEMORY_TOKEN_LIMIT,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
)

# Load environment variables from .env FIRST, before any function reads them
load_dotenv()


def _get_groq_api_key() -> str:
    """Helper to fetch and validate the Groq API key from .env."""
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file."
        )
    return api_key


def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model."""
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )


def initialise_llm(model_name: str = None) -> Groq:
    """
    Initialises the Groq LLM with core parameters from config.
    If model_name is provided, uses that instead of the default LLM_MODEL.
    """
    # Use the provided model name or fall back to the default
    model_to_use = model_name or LLM_MODEL
    
    return Groq(
        api_key=_get_groq_api_key(),
        model=model_to_use,
        max_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
    )


def initialise_hyde_llm() -> Groq:
    """
    Initialises a faster Groq LLM specifically for query transformation (HyDE).
    Using a smaller model ensures the 'imagination' step is fast and quota-efficient.
    """
    return Groq(
        api_key=_get_groq_api_key(),
        model="llama-3.1-8b-instant",
        max_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
