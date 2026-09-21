# =====================================================
# FASTAPI SERVER BACKEND (REST API FOR LECTURA AI)
# =====================================================

import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Import core modules
from utils.audio_processor import download_youtube_audio, convert_to_wav, split_audio
from core.transcriber import run_transcription
from core.document_creator import create_documents
from core.vector_store import build_vector_store
from core.summarizer import generate_summary
from core.rag_engine import ask_question, format_timestamp
from core.task_extractor import extract_questions_and_tasks
from utils.exporter import generate_pdf_notes, generate_txt_export

load_dotenv()

app = FastAPI(title="Lectura AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
os.makedirs("downloads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static folder for media & assets
app.mount("/downloads", StaticFiles(directory="downloads"), name="downloads")

# Global state tracker
workspace_state = {
    "youtube_url": "https://youtu.be/XN3xNJvWXsc?si=3BKO-0-kD-eylrFEW",
    "media_file_path": None,
    "chat_history": [],
    "is_processing": False,
    "processing_step": "Idle",
    "error_message": None
}

def run_ingest_pipeline(audio_file: Optional[str], youtube_url: Optional[str], model_name: str, task: str, mistral_key: str):
    try:
        workspace_state["is_processing"] = True
        workspace_state["error_message"] = None

        if youtube_url and not audio_file:
            workspace_state["processing_step"] = "Downloading YouTube Audio..."
            audio_file = download_youtube_audio(youtube_url, output_dir="downloads")

        workspace_state["processing_step"] = "Converting & Splitting Audio Chunks..."
        wav_file = convert_to_wav(audio_file)
        split_audio(wav_file, output_folder="downloads/audio_chunks", chunk_minutes=10)

        workspace_state["processing_step"] = "Transcribing with Whisper AI..."
        trans_segments = run_transcription(
            chunks_dir="downloads/audio_chunks",
            output_file="data/transcription.json",
            model_name=model_name,
            task=task
        )

        workspace_state["processing_step"] = "Building Vector Database..."
        docs = create_documents(input_file="data/transcription.json", output_file="data/documents.json")
        build_vector_store(documents_input=docs)

        workspace_state["processing_step"] = "Generating AI Summary..."
        summary_txt = generate_summary(
            transcription_input=trans_segments,
            output_file="data/summary.txt",
            mistral_api_key=mistral_key
        )

        workspace_state["processing_step"] = "Extracting Questions & Study Tasks..."
        extracted_tasks = extract_questions_and_tasks(
            transcription_input=trans_segments,
            mistral_api_key=mistral_key
        )
        with open("data/extracted_tasks.json", "w", encoding="utf-8") as f:
            json.dump(extracted_tasks, f, indent=4)

        workspace_state["is_processing"] = False
        workspace_state["processing_step"] = "Completed"
    except Exception as e:
        workspace_state["is_processing"] = False
        workspace_state["processing_step"] = "Error"
        workspace_state["error_message"] = str(e)

class IngestRequest(BaseModel):
    youtube_url: Optional[str] = None
    model_name: Optional[str] = "base"
    task: Optional[str] = "transcribe"
    api_key: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 2
    api_key: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Lectura AI API Server is Running</h1>")

@app.get("/api/status")
async def get_status():
    has_db = os.path.exists("data/vector_db")
    has_trans = os.path.exists("data/transcription.json")
    has_sum = os.path.exists("data/summary.txt")
    has_tasks = os.path.exists("data/extracted_tasks.json")
    
    num_segments = 0
    if has_trans:
        try:
            with open("data/transcription.json", "r", encoding="utf-8") as f:
                num_segments = len(json.load(f))
        except Exception:
            pass

    return {
        "has_vector_db": has_db,
        "has_transcription": has_trans,
        "has_summary": has_sum,
        "has_tasks": has_tasks,
        "num_segments": num_segments,
        "youtube_url": workspace_state["youtube_url"],
        "media_file_path": workspace_state["media_file_path"],
        "has_mistral_key": bool(os.getenv("MISTRAL_API_KEY")),
        "is_processing": workspace_state.get("is_processing", False),
        "processing_step": workspace_state.get("processing_step", "Idle"),
        "error_message": workspace_state.get("error_message")
    }

@app.post("/api/ingest")
async def ingest_media(
    background_tasks: BackgroundTasks,
    youtube_url: Optional[str] = Form(None),
    model_name: Optional[str] = Form("base"),
    task: Optional[str] = Form("transcribe"),
    api_key: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=400, detail="Mistral API Key is required.")

    if workspace_state.get("is_processing"):
        raise HTTPException(status_code=400, detail="Processing is already running in background. Please wait.")

    try:
        audio_file = None
        if file:
            save_path = os.path.join("downloads", file.filename)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            audio_file = save_path
            workspace_state["youtube_url"] = ""
            workspace_state["media_file_path"] = f"/downloads/{file.filename}"
        elif youtube_url:
            workspace_state["youtube_url"] = youtube_url
            workspace_state["media_file_path"] = None
        else:
            raise HTTPException(status_code=400, detail="Please provide a YouTube URL or upload a file.")

        background_tasks.add_task(
            run_ingest_pipeline,
            audio_file=audio_file,
            youtube_url=youtube_url if not audio_file else None,
            model_name=model_name,
            task=task,
            mistral_key=mistral_key
        )

        return {
            "status": "processing",
            "message": "Processing started in background!",
            "youtube_url": workspace_state["youtube_url"],
            "media_file_path": workspace_state["media_file_path"]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_lecture(req: ChatRequest):
    if req.api_key:
        os.environ["MISTRAL_API_KEY"] = req.api_key

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=400, detail="Mistral API Key is required.")

    if not os.path.exists("data/vector_db"):
        raise HTTPException(status_code=400, detail="No processed lecture found. Ingest a video first.")

    try:
        answer, sources = ask_question(
            question=req.question,
            chroma_dir="data/vector_db",
            collection_name="youtube_transcript",
            mistral_api_key=mistral_key,
            top_k=req.top_k
        )
        
        workspace_state["chat_history"].append({"role": "user", "content": req.question})
        workspace_state["chat_history"].append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summary")
async def get_summary():
    if os.path.exists("data/summary.txt"):
        with open("data/summary.txt", "r", encoding="utf-8") as f:
            return {"summary": f.read()}
    return {"summary": "No summary generated yet."}

@app.get("/api/transcript")
async def get_transcript():
    if os.path.exists("data/transcription.json"):
        with open("data/transcription.json", "r", encoding="utf-8") as f:
            return {"transcript": json.load(f)}
    return {"transcript": []}

@app.get("/api/tasks")
async def get_tasks():
    if os.path.exists("data/extracted_tasks.json"):
        with open("data/extracted_tasks.json", "r", encoding="utf-8") as f:
            return {"tasks": json.load(f)}
    return {"tasks": []}

@app.post("/api/tasks/reextract")
async def reextract_tasks(api_key: Optional[str] = Form(None)):
    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=400, detail="Mistral API Key is required.")

    if not os.path.exists("data/transcription.json"):
        raise HTTPException(status_code=400, detail="No transcription found. Ingest a video first.")

    try:
        extracted_tasks = extract_questions_and_tasks(
            transcription_input="data/transcription.json",
            mistral_api_key=mistral_key
        )
        with open("data/extracted_tasks.json", "w", encoding="utf-8") as f:
            json.dump(extracted_tasks, f, indent=4)
        return {"status": "success", "tasks": extracted_tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/pdf")
async def export_pdf():
    summary_str = ""
    if os.path.exists("data/summary.txt"):
        with open("data/summary.txt", "r", encoding="utf-8") as f:
            summary_str = f.read()

    pdf_path = generate_pdf_notes(
        video_title=workspace_state["youtube_url"] or "Uploaded Lecture File",
        summary_text=summary_str,
        chat_history=workspace_state["chat_history"],
        output_path="downloads/Lecture_Notes.pdf"
    )
    return FileResponse(pdf_path, media_type="application/pdf", filename="Lecture_Notes.pdf")

@app.get("/api/export/txt")
async def export_txt():
    t_data = []
    if os.path.exists("data/transcription.json"):
        with open("data/transcription.json", "r", encoding="utf-8") as f:
            t_data = json.load(f)
    sum_str = ""
    if os.path.exists("data/summary.txt"):
        with open("data/summary.txt", "r", encoding="utf-8") as f:
            sum_str = f.read()

    txt_content = generate_txt_export(sum_str, t_data, workspace_state["chat_history"])
    return HTMLResponse(content=txt_content, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)


