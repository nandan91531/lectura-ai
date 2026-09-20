
# ============================================================
# AUDIO PROCESSOR UTILITIES
# ============================================================

import os
import glob
from pathlib import Path
from pydub import AudioSegment
import yt_dlp

def download_youtube_audio(url, output_dir="downloads"):
    """
    Downloads best audio from a YouTube URL into the output directory.
    Returns path to downloaded audio file.
    """
    download_folder = Path(output_dir)
    download_folder.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(download_folder / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)

    return downloaded_file

def convert_to_wav(input_path, output_path=None):
    """
    Converts input audio/video file to 16kHz mono WAV suitable for Whisper.
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path

def split_audio(input_path, output_folder=None, chunk_minutes=10, progress_callback=None):
    """
    Splits WAV audio into chunks of `chunk_minutes` and saves in `output_folder`.
    Returns list of chunk file paths.
    """
    audio = AudioSegment.from_file(input_path)
    chunk_length = chunk_minutes * 60 * 1000  # ms
    
    if output_folder is None:
        output_folder = Path(input_path).parent / "audio_chunks"
    else:
        output_folder = Path(output_folder)
        
    # Clean previous chunks if any
    output_folder.mkdir(parents=True, exist_ok=True)
    for old_file in glob.glob(str(output_folder / "*.wav")):
        try:
            os.remove(old_file)
        except Exception:
            pass

    chunk_files = []
    total_chunks = (len(audio) + chunk_length - 1) // chunk_length

    for i, start in enumerate(range(0, len(audio), chunk_length)):
        end = min(start + chunk_length, len(audio))
        chunk = audio[start:end]
        
        chunk_filename = output_folder / f"chunk_{i + 1:03d}.wav"
        chunk.export(chunk_filename, format="wav")
        chunk_files.append(str(chunk_filename))

        if progress_callback:
            progress_callback(i + 1, total_chunks, f"Chunk {i + 1}/{total_chunks} created")

    return chunk_files