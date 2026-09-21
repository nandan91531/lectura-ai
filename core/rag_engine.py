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
    Converts seconds float into MM:SS or HH:MM:SS format.
    """
    if seconds is None:
        return "00:00"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    rem_seconds = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{rem_seconds:02d}"
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
    top_k=2,
    temperature=0.2
):
    """
    Performs vector similarity search on ChromaDB and generates AI answer with timestamp sources.
    Returns: (answer_text, sources_list) - sources_list contains at most top 2 relevant timestamps.
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

    # Use k=top_k (max 2 by default)
    k_retrieve = min(top_k, 2) if top_k else 2
    results = vector_store.similarity_search(question, k=k_retrieve)

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

    # Strictly limit to top 2 sources maximum
    sources = sources[:2]

    context_str = "\n\n".join(context_parts)

    llm = ChatMistralAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        max_tokens=2048
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant for lecture videos.

Answer the user's question directly, concisely, and accurately using ONLY the information provided in the lecture context.

STRICT RULES:
1. STRICT CONTEXT GROUNDING: Rely ONLY on facts explicitly stated in the provided lecture context. Never assume, extrapolate, or bring in outside general knowledge not present in the lecture.
2. ANSWER ONLY WHAT IS ASKED: Be direct and concise. Answer only the specific question asked by the user. Do NOT dump unrelated lecture content, definitions, or full lecture summaries unless explicitly requested.
3. ABSENCE OF INFORMATION: If the answer to the user's specific question is not present in the provided lecture context, state clearly and concisely: "This topic was not covered in the lecture video."
4. NO FILLER INTROS: Start IMMEDIATELY with the answer. NEVER use conversational preambles or filler intros (e.g. NEVER say "Based on the provided snippets...", "Here is the answer:", "According to the lecture...").
5. FORMATTING: Use clean, concise standard Markdown with clear bullet points or short paragraphs where appropriate.
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