# =====================================================
# VECTOR STORE MODULE (CHROMADB + HUGGINGFACE)
# =====================================================

import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "data/vector_db"
COLLECTION_NAME = "youtube_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings_model(model_name=EMBEDDING_MODEL):
    """
    Returns HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def build_vector_store(
    documents_input="data/documents.json",
    chroma_dir=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_model=EMBEDDING_MODEL,
    reset=True
):
    """
    Reads document chunks, embeds them, and stores them in ChromaDB.
    Resets existing vector database if reset=True.
    """
    if isinstance(documents_input, str):
        if not os.path.exists(documents_input):
            raise FileNotFoundError(f"Documents file not found: {documents_input}")
        with open(documents_input, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = documents_input

    documents = [
        Document(
            page_content=item["text"],
            metadata=item["metadata"]
        )
        for item in data
    ]

    embeddings = get_embeddings_model(embedding_model)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir
    )

    if reset:
        try:
            existing_ids = vector_store.get()["ids"]
            if existing_ids:
                vector_store.delete(ids=existing_ids)
        except Exception:
            pass

    vector_store.add_documents(documents)
    return vector_store


if __name__ == "__main__":
    build_vector_store()