
# =====================================================
# RAG CHATBOT: CHROMADB + MISTRAL API
# =====================================================

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# 1. CONFIGURATION
# =====================================================

CHROMA_DIR = "data/vector_db"
COLLECTION_NAME = "youtube_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = "ministral-3b-2512"


# =====================================================
# 2. LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise ValueError(
        "MISTRAL_API_KEY nahi mili. Apni .env file check karo."
    )


# =====================================================
# 3. FORMAT TIMESTAMP
# =====================================================

def format_timestamp(seconds):
    """
    Seconds ko minutes aur seconds mein convert karta hai.
    Example: 378.4 -> 6 min 18 sec
    """

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes} min {remaining_seconds} sec"


# =====================================================
# 4. LOAD EMBEDDING MODEL
# =====================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded successfully!")


# =====================================================
# 5. CONNECT TO CHROMADB
# =====================================================

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

print("Connected to ChromaDB successfully!")


# =====================================================
# 6. CONNECT TO MISTRAL API
# =====================================================

llm = ChatMistralAI(
    model=MISTRAL_MODEL,
    temperature=0.2
)

print("Connected to Mistral API successfully!")


# =====================================================
# 7. CREATE RAG PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an intelligent YouTube lecture assistant.

Answer the user's question directly, concisely, and accurately using ONLY the provided context.

Rules:
1. STRICT CONTEXT GROUNDING: Do not invent or assume information outside the provided context.
2. ANSWER ONLY WHAT IS ASKED: Be direct and concise. Answer only the specific question asked by the user without dumping full lecture summaries.
3. ABSENCE OF INFORMATION: If the answer is not present in the provided context, state clearly: "This topic was not covered in the lecture."
4. NO FILLER INTROS: Start immediately with the answer without preambles or filler intros.
5. Mention approximate timestamps when relevant.

Context:
{context}

User Question:
{question}

Answer:
"""
)


# =====================================================
# 8. CHAT LOOP
# =====================================================

while True:

    question = input(
        "\nAsk your question (type exit to stop): "
    ).strip()

    if question.lower() == "exit":
        print("Chatbot closed!")
        break

    if not question:
        print("Please enter a question.")
        continue

    print("\nSearching relevant lecture content...")

    # Relevant chunks retrieve karo
    results = vector_store.similarity_search(
        question,
        k=1
    )

    if not results:
        print("No relevant content found.")
        continue

    # Retrieved documents ko context mein convert karo
    context_parts = []

    for document in results:

        start_seconds = document.metadata.get("start", 0)
        end_seconds = document.metadata.get("end", 0)

        context_parts.append(
            f"""
Timestamp: {format_timestamp(start_seconds)}
to {format_timestamp(end_seconds)}

Content:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)


    # =================================================
    # 9. SEND CONTEXT TO MISTRAL
    # =================================================

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    print("Generating answer...")

    response = llm.invoke(formatted_prompt)


    # =================================================
    # 10. DISPLAY FINAL ANSWER
    # =================================================

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.content)


    # Existing timestamp database se read karo
    document = results[0]

    start = document.metadata.get("start", 0)
    end = document.metadata.get("end", 0)

    print("\n📌 Lecture Timestamp:")
    print(f"Start: {format_timestamp(start)}")
    print(f"End: {format_timestamp(end)}")

    print("=" * 60)