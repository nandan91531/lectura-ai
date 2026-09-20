# =====================================================
# RAG ENGINE MODULE (CHROMADB + MISTRAL API)
# =====================================================

import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "data/vector_db"
COLLECTION_NAME = "youtube_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = "ministral-3b-2512"

def format_timestamp(seconds):
    """
    Converts seconds float into MM:SS format.
    """
    minutes = int(seconds // 60)
    rem_seconds = int(seconds % 60)
    return f"{minutes:02d}:{rem_seconds:02d}"

def clean_llm_response(text: str) -> str:
    """Clean up conversational filler prefixes and ensure markdown headers/lists have proper newlines."""
    if not text:
        return ""
    
    # Remove common preamble intros like "Here's a structured breakdown..." or "Based on the provided lecture snippets..."
    text = re.sub(r'^(Here\'s|Here is|Based on the provided|Below is|According to the).*?:\s*', '', text, flags=re.IGNORECASE)
    # Remove standalone horizontal rules or "Answer:" headers at the very start
    text = re.sub(r'^\s*---\s*', '', text)
    text = re.sub(r'^\s*###?\s*\*\*Answer.*?\*\*\s*:?\s*', '', text, flags=re.IGNORECASE)
    
    # Ensure headings have blank lines before them
    text = re.sub(r'([^\n])\n(#{1,6}\s+)', r'\1\n\n\2', text)
    # Ensure list items have proper newline spacing from preceding text
    text = re.sub(r'([^\n])\n([*\-]\s+)', r'\1\n\n\2', text)
    
    return text.strip()

def ask_question(
    question,
    chroma_dir=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_model=EMBEDDING_MODEL,
    mistral_api_key=None,
    model_name=MISTRAL_MODEL,
    top_k=4,
    temperature=0.2
):
    """
    Performs vector similarity search on ChromaDB and generates AI answer with timestamp sources.
    Returns: (answer_text, sources_list)
    """
    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API Key is missing. Provide it in .env or via settings sidebar.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if not os.path.exists(chroma_dir):
        return "Vector database not found. Please upload or ingest a lecture video/audio first.", []

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir
    )

    results = vector_store.similarity_search(question, k=top_k)

    if not results:
        return "No relevant information found in the video transcript for your question.", []

    context_parts = []
    sources = []
    seen_timestamps = set()

    for idx, doc in enumerate(results):
        start = doc.metadata.get("start", 0)
        end = doc.metadata.get("end", 0)
        start_fmt = format_timestamp(start)
        end_fmt = format_timestamp(end)

        context_parts.append(
            f"--- Source [{idx + 1}] (Timestamp: {start_fmt} to {end_fmt}) ---\n{doc.page_content}"
        )

        ts_key = (round(start, 2), round(end, 2))
        if ts_key not in seen_timestamps:
            seen_timestamps.add(ts_key)
            sources.append({
                "start": start,
                "end": end,
                "start_fmt": start_fmt,
                "end_fmt": end_fmt,
                "content": doc.page_content
            })

    context_str = "\n\n".join(context_parts)

    llm = ChatMistralAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        max_tokens=2048
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant for lecture videos.

Answer the user's question accurately, completely, and directly using ALL relevant information from the provided lecture snippets.

STRICT RULES:
1. Start IMMEDIATELY with the answer. NEVER use conversational preambles or filler intros (e.g. NEVER say "Here's a structured breakdown...", "Based on the provided snippets...", "Here is the answer:").
2. Ensure the answer is COMPLETE and covers ALL types, categories, steps, definitions, and examples mentioned across the provided context without omitting items.
3. Use clean, standard Markdown with double line breaks between paragraphs, headings, and list items.
4. Keep subheadings brief and bold key terms.
5. Do NOT invent facts outside the context. If the topic is missing, state: "This topic was not covered in the lecture."
"""),
        ("human", """Context from Lecture:
{context}

User Question:
{question}""")
    ])

    formatted_messages = prompt.format_messages(context=context_str, question=question)
    response = llm.invoke(formatted_messages)

    clean_answer = clean_llm_response(response.content)

    return clean_answer, sources