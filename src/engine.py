from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import BaseRetriever, TransformRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
    RERANKER_TOP_N,
    RERANKER_MODEL_NAME,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
    initialise_hyde_llm,
)

def _create_new_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""
    print("Creating new vector store from all files in the 'data' directory...")
    documents: list[Document] = SimpleDirectoryReader(input_dir=DATA_PATH).load_data()

    if not documents:
        raise ValueError(f"No documents found in {DATA_PATH}. Cannot create vector store.")

    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )

    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index

def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Loads the vector store from disk if it exists; otherwise, creates a new one."""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        return _create_new_vector_store(embed_model)

def get_chat_engine(llm: Groq, embed_model: HuggingFaceEmbedding) -> CondensePlusContextChatEngine:
    """Initialises and returns the main conversational RAG chat engine with HyDE."""
    vector_index: VectorStoreIndex = get_vector_store(embed_model)

    base_retriever: BaseRetriever = vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

    hyde = HyDEQueryTransform(include_original=True, llm=llm)
    hyde_retriever = TransformRetriever(retriever=base_retriever, query_transform=hyde)

    reranker = SentenceTransformerRerank(top_n=RERANKER_TOP_N, model=RERANKER_MODEL_NAME)
    
    memory = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT,
        llm=llm
    )

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=hyde_retriever,
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
        node_postprocessors=[reranker]
    )
    return chat_engine


# ==========================================
# NEW: Web-friendly Wrapper Class
# ==========================================
class RAGPipeline:
    """Wraps the RAG logic for use in Streamlit (prevents reloading models on every message)."""
    
    def __init__(self):
        print("--- Initialising models for Web... ---")
        self.llm = initialise_llm()
        self.embed_model = get_embedding_model()
        self.chat_engine = get_chat_engine(llm=self.llm, embed_model=self.embed_model)
        print("--- RAG Chatbot Initialised for Web. ---")

    def ask(self, prompt: str) -> tuple[str, list[str]]:
        """
        Takes a prompt, returns the response string and a list of source chunks.
        """
        response = self.chat_engine.chat(prompt)
        answer = str(response)
        
        # Extract real sources from LlamaIndex response nodes
        sources = []
        if response.source_nodes:
            for node in response.source_nodes:
                # Try to get the filename, fallback to generic name
                file_name = node.metadata.get('file_name', 'Unknown Document')
                chunk_text = node.text[:150].replace("\n", " ") + "..."
                sources.append(f"📄 **{file_name}**: {chunk_text}")
                
        return answer, sources


# ==========================================
# ORIGINAL: Terminal REPL Loop
# ==========================================
def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot in the terminal."""
    print("--- Initialising models... ---")
    llm: Groq = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: BaseChatEngine = get_chat_engine(llm=llm, embed_model=embed_model)
    print("--- RAG Chatbot Initialised. ---")
    
    # Note: chat_repl() is a built-in LlamaIndex method for terminal UIs
    chat_engine.chat_repl()

if __name__ == "__main__":
    main_chat_loop()
