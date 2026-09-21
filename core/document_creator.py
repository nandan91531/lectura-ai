# =====================================================
# DOCUMENT CREATOR MODULE
# =====================================================

import json
import os

def create_documents(
    input_file="data/transcription.json",
    output_file="data/documents.json",
    max_chunk_size=250,
    overlap_segments=1
):
    """
    Groups Whisper segments into structured documents with overlap and timestamp metadata.
    Uses a smaller chunk size (max_chunk_size=250) so timestamps pinpoint exact video locations.
    """
    if isinstance(input_file, str):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        segments = input_file  # List of dicts directly passed

    documents = []
    current_segments = []
    current_length = 0

    for segment in segments:
        text = segment["text"]
        current_segments.append(segment)
        current_length += len(text)

        if current_length >= max_chunk_size:
            chunk_text = " ".join(item["text"] for item in current_segments)
            document = {
                "text": chunk_text,
                "metadata": {
                    "start": float(current_segments[0]["start"]),
                    "end": float(current_segments[-1]["end"])
                }
            }
            documents.append(document)

            current_segments = current_segments[-overlap_segments:] if overlap_segments > 0 else []
            current_length = sum(len(item["text"]) for item in current_segments)

    if current_segments:
        chunk_text = " ".join(item["text"] for item in current_segments)
        document = {
            "text": chunk_text,
            "metadata": {
                "start": float(current_segments[0]["start"]),
                "end": float(current_segments[-1]["end"])
            }
        }
        documents.append(document)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=4, ensure_ascii=False)

    return documents

if __name__ == "__main__":
    create_documents()