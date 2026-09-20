# =====================================================
# QUIZ & FLASHCARD GENERATOR MODULE (MISTRAL AI)
# =====================================================

import json
import os
import re
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_quiz(
    transcription_input="data/transcription.json",
    mistral_api_key=None,
    model_name="ministral-3b-2512",
    num_questions=5
):
    """
    Generates interactive MCQ quiz questions based on lecture transcript using Mistral AI.
    Returns list of dicts: [{"id": 1, "question": "...", "options": [...], "answer": "...", "explanation": "..."}]
    """
    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API Key is required for Quiz Generation.")

    if isinstance(transcription_input, str):
        if not os.path.exists(transcription_input):
            return []
        with open(transcription_input, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = transcription_input

    full_text = " ".join(doc.get("text", "") for doc in data)[:12000] # truncate if extremely long

    llm = ChatMistralAI(model=model_name, temperature=0.3, api_key=api_key)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert educator. Create exactly {num_questions} multiple-choice quiz questions (MCQs) based ONLY on this lecture content to test student understanding.

CRITICAL INSTRUCTION: Return ONLY a raw valid JSON array. Do not include markdown ticks, no ```json formatting, no commentary.

Format required:
[
  {{
    "id": 1,
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "answer": "B) Option 2",
    "explanation": "Detailed explanation of why this answer is correct."
  }}
]

Lecture Content:
{text}
"""
    )

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({"num_questions": num_questions, "text": full_text})

    # Clean potential markdown wrapping
    cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()
    
    try:
        quiz_data = json.loads(cleaned)
        return quiz_data
    except Exception as e:
        # Fallback parsing
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return []

def generate_flashcards(
    transcription_input="data/transcription.json",
    mistral_api_key=None,
    model_name="ministral-3b-2512"
):
    """
    Generates study flashcards (Term & Definition) from lecture text.
    Returns list of dicts: [{"term": "...", "definition": "..."}]
    """
    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return []

    if isinstance(transcription_input, str):
        if not os.path.exists(transcription_input):
            return []
        with open(transcription_input, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = transcription_input

    full_text = " ".join(doc.get("text", "") for doc in data)[:10000]

    llm = ChatMistralAI(model=model_name, temperature=0.2, api_key=api_key)

    prompt = ChatPromptTemplate.from_template(
        """Extract 5 important key terms, acronyms, or concepts from this lecture and provide concise definitions for revision.

Return ONLY a raw valid JSON array:
[
  {{
    "term": "Term Name",
    "definition": "Clear concise 1-2 sentence definition."
  }}
]

Lecture Content:
{text}
"""
    )

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({"text": full_text})
    cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return []
