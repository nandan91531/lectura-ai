# =====================================================
# TASK & QUESTION EXTRACTOR MODULE (MISTRAL AI)
# =====================================================

import json
import os
import re
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def extract_questions_and_tasks(
    transcription_input="data/transcription.json",
    mistral_api_key=None,
    model_name="ministral-3b-2512"
):
    """
    Extracts all questions asked by the instructor, practice problems, homework tasks, 
    and exercise prompts from the lecture transcript with approximate timestamps.
    Returns list of dicts: [{"type": "Homework / Question", "timestamp": "04:15", "task": "...", "context": "..."}]
    """
    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return []

    if isinstance(transcription_input, str):
        if not os.path.exists(transcription_input):
            return []
        with open(transcription_input, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        segments = transcription_input

    # Prepare timestamped text chunks
    formatted_transcript_lines = []
    for seg in segments:
        s_min = int(seg.get("start", 0) // 60)
        s_sec = int(seg.get("start", 0) % 60)
        ts_str = f"[{s_min:02d}:{s_sec:02d}]"
        formatted_transcript_lines.append(f"{ts_str} {seg.get('text', '')}")

    full_formatted_text = "\n".join(formatted_transcript_lines)[:14000]

    llm = ChatMistralAI(model=model_name, temperature=0.1, api_key=api_key)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert academic assistant. Scan the timestamped lecture transcript below and extract ALL questions asked by the speaker, practice problems, homework tasks, exercises, code challenges, or key discussion questions.

If no explicit homework is assigned, extract 3 to 6 key self-assessment questions and discussion points that test understanding of the material along with their approximate timestamps.

Return ONLY a raw valid JSON array. Do not include markdown code block formatting (` ```json ` or ` ``` `).

JSON Schema required:
[
  {{
    "timestamp": "MM:SS",
    "type": "Question / Homework / Exercise / Discussion",
    "task": "The specific question or task asked or implied in the lecture",
    "explanation": "Brief context or answer guidance based on the video"
  }}
]

Timestamped Transcript:
{transcript}
"""
    )

    chain = prompt | llm | StrOutputParser()
    try:
        raw_response = chain.invoke({"transcript": full_formatted_text})
    except Exception as e:
        print(f"LLM task extraction error: {e}")
        return []

    cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()

    # Attempt 1: Direct JSON parsing
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Attempt 2: Extract JSON array with regex
    match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # Attempt 3: Heuristic line parsing fallback
    fallback_tasks = []
    lines = cleaned.split("\n")
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        ts_match = re.search(r"\[?(\d{1,2}:\d{2})\]?", line_str)
        timestamp = ts_match.group(1) if ts_match else "00:00"
        
        if "question" in line_str.lower() or "?" in line_str:
            task_type = "Question"
        elif "homework" in line_str.lower() or "task" in line_str.lower():
            task_type = "Homework"
        else:
            task_type = "Discussion"

        clean_text = re.sub(r"^[-\*\d\.\s]+", "", line_str)
        if len(clean_text) > 10:
            fallback_tasks.append({
                "timestamp": timestamp,
                "type": task_type,
                "task": clean_text[:120],
                "explanation": "Extracted from lecture discussion."
            })

    return fallback_tasks

