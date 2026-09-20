# =====================================================
# TRANSCRIBER MODULE (OPENAI WHISPER)
# =====================================================

import os
import glob
import json
import whisper

def run_transcription(
    chunks_dir="downloads/audio_chunks",
    output_file="data/transcription.json",
    model_name="small",
    task="transcribe",
    language=None,
    progress_callback=None
):
    """
    Transcribes all .wav chunks in chunks_dir using OpenAI Whisper.
    Saves clean timestamped segments to output_file and returns the segments list.
    """
    files = sorted(glob.glob(os.path.join(chunks_dir, "*.wav")))
    if not files:
        raise FileNotFoundError(f"No WAV chunks found in directory: {chunks_dir}")

    if progress_callback:
        progress_callback(0, len(files), f"Loading Whisper model '{model_name}'...")

    model = whisper.load_model(model_name)
    all_segments = []

    for index, file_path in enumerate(files):
        if progress_callback:
            progress_callback(index, len(files), f"Transcribing chunk {index + 1}/{len(files)}...")

        transcribe_kwargs = {"fp16": False, "task": task}
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        result = model.transcribe(file_path, **transcribe_kwargs)
        chunk_offset = index * 600  # 10 minutes chunk offset

        for segment in result["segments"]:
            clean_segment = {
                "start": round(segment["start"] + chunk_offset, 2),
                "end": round(segment["end"] + chunk_offset, 2),
                "text": segment["text"].strip()
            }
            all_segments.append(clean_segment)

    if progress_callback:
        progress_callback(len(files), len(files), "Saving transcription JSON...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=4, ensure_ascii=False)

    return all_segments

if __name__ == "__main__":
    run_transcription()