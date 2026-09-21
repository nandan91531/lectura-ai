
# =====================================================
# SUMMARIZER MODULE (MISTRAL AI)
# =====================================================

import json
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_summary(
    transcription_input="data/transcription.json",
    output_file="data/summary.txt",
    mistral_api_key=None,
    model_name="ministral-3b-2512",
    progress_callback=None
):
    """
    Generates a concise final summary from transcription segments using Mistral AI.
    """
    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API Key is required. Please set MISTRAL_API_KEY in .env or via settings.")

    if isinstance(transcription_input, str):
        if not os.path.exists(transcription_input):
            raise FileNotFoundError(f"Transcription file not found: {transcription_input}")
        with open(transcription_input, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        documents = transcription_input

    full_text = " ".join(doc.get("text", "") for doc in documents)
    if not full_text.strip():
        return "No text available to summarize."

    # 1. Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    # 2. Initialize Mistral LLM
    llm = ChatMistralAI(
        model=model_name,
        temperature=0.2,
        api_key=api_key
    )

    # 3. Chunk summary chain
    chunk_prompt = ChatPromptTemplate.from_template(
        """Summarize this transcript section concisely in bullet points or short paragraph.

Rules:
- Use clear English
- Highlight essential key concepts, definitions, and examples
- Avoid fluff and repetition
- Maximum 120 words per section

Transcript Section:
{text}
"""
    )
    chunk_chain = chunk_prompt | llm | StrOutputParser()

    summaries = []
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(idx, total_chunks, f"Summarizing section {idx + 1}/{total_chunks}...")
        summary = chunk_chain.invoke({"text": chunk})
        summaries.append(summary)

    combined_summaries = "\n\n".join(summaries)

    # 4. Final synthesis prompt
    final_prompt = ChatPromptTemplate.from_template(
        """Write a comprehensive and well-structured study summary of the video lecture.

Structure your response with:
1. Executive Overview (2-3 sentences starting with "This video explains...")
2. Key Topics & Core Concepts (bullet points with short explanations)
3. Takeaways & Conclusions

Rules:
- Use clear, natural language
- Be detailed yet easy to digest for students
- Do not mention 'chunks' or 'transcripts'

Video Content:
{text}
"""
    )
    final_chain = final_prompt | llm | StrOutputParser()

    if progress_callback:
        progress_callback(total_chunks, total_chunks, "Generating final synthesized summary...")

    final_summary = final_chain.invoke({"text": combined_summaries})

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_summary)

    return final_summary

if __name__ == "__main__":
    generate_summary()
