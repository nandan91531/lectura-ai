import os
import json
from pathlib import Path
import streamlit as st
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

# =====================================================
# PAGE CONFIG & CLEAN PREMIUM STYLING
# =====================================================
st.set_page_config(
    page_title="Lectura AI - YouTube & Video Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# Inject Clean Custom CSS for Balanced Spacing & Hierarchy
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Container Constrain */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    /* App Header */
    .nav-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Hero Input Box */
    .hero-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 24px;
    }

    /* Clean Card Boxes */
    .info-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .card-heading {
        font-size: 1rem;
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 8px;
    }

    /* Timestamp Pill */
    .ts-pill {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: monospace;
        display: inline-block;
        margin-right: 6px;
    }

    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
        border-color: #6366f1 !important;
    }

    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = "https://youtu.be/XN3xNJvWXsc?si=3BKO-0-kD-eylrFEW"
if "video_file_path" not in st.session_state:
    st.session_state.video_file_path = None
if "extracted_tasks" not in st.session_state:
    st.session_state.extracted_tasks = []

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    env_api_key = os.getenv("MISTRAL_API_KEY", "")
    api_key_input = st.text_input(
        "Mistral API Key",
        value=env_api_key,
        type="password",
        placeholder="Paste API Key..."
    )
    if api_key_input:
        os.environ["MISTRAL_API_KEY"] = api_key_input
        st.caption("✅ API Key active")
    else:
        st.caption("⚠️ API Key required")

    st.markdown("---")
    whisper_model = st.selectbox("Whisper Accuracy", ["tiny", "base", "small", "medium"], index=2)
    whisper_task = st.selectbox("Language Mode", ["transcribe", "translate"], format_func=lambda x: "Original Language" if x == "transcribe" else "Translate to English")
    top_k_chunks = st.slider("Context Chunks", 1, 10, 4)

    st.markdown("---")
    if st.button("🗑️ Reset Workspace", use_container_width=True):
        st.session_state.messages = []
        st.session_state.extracted_tasks = []
        st.rerun()

# =====================================================
# NAVBAR
# =====================================================
st.markdown("""
<div class="nav-header">
    <div>
        <div class="brand-title">🎓 Lectura AI — Video Intelligence Assistant</div>
        <div class="brand-sub">Transcribe, summarize, extract tasks & chat with any lecture video</div>
    </div>
</div>
""", unsafe_allow_html=True)

# File status checks
has_vector_db = os.path.exists("data/vector_db")
has_transcription = os.path.exists("data/transcription.json")
has_summary = os.path.exists("data/summary.txt")

# Load existing extracted tasks
if not st.session_state.extracted_tasks and os.path.exists("data/extracted_tasks.json"):
    try:
        with open("data/extracted_tasks.json", "r", encoding="utf-8") as f:
            st.session_state.extracted_tasks = json.load(f)
    except Exception:
        pass

# =====================================================
# HERO MEDIA INPUT & PLAYER SECTION
# =====================================================
with st.container():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    
    c_source, c_input, c_action = st.columns([1.5, 3, 1])
    
    with c_source:
        source_mode = st.radio("Media Source:", ["YouTube URL", "Upload Video/Audio File"], label_visibility="collapsed")

    if source_mode == "YouTube URL":
        with c_input:
            input_url = st.text_input(
                "YouTube URL",
                value=st.session_state.youtube_url,
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed"
            )
        with c_action:
            process_click = st.button("⚡ Process Video", type="primary", use_container_width=True)
    else:
        with c_input:
            uploaded_file = st.file_uploader("Upload Video/Audio File", type=["mp4", "mp3", "m4a", "wav"], label_visibility="collapsed")
            if uploaded_file:
                os.makedirs("downloads", exist_ok=True)
                uploaded_path = os.path.join("downloads", uploaded_file.name)
                with open(uploaded_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.video_file_path = uploaded_path
        with c_action:
            process_click = st.button("⚡ Process Video", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Processing Trigger Execution
if process_click:
    if not api_key_input:
        st.error("Please enter your Mistral API Key in the sidebar settings!")
    else:
        status = st.empty()
        progress = st.progress(0)
        
        try:
            if source_mode == "YouTube URL":
                st.session_state.youtube_url = input_url
                st.session_state.video_file_path = None
                status.info("1/5 Downloading audio from YouTube...")
                progress.progress(15)
                audio_file = download_youtube_audio(input_url, output_dir="downloads")
            else:
                if 'uploaded_path' not in locals() or not uploaded_path:
                    st.error("Please select a file to upload first!")
                    st.stop()
                st.session_state.youtube_url = ""
                audio_file = uploaded_path

            status.info("2/5 Converting audio to 16kHz WAV format...")
            progress.progress(35)
            wav_file = convert_to_wav(audio_file)
            split_audio(wav_file, output_folder="downloads/audio_chunks", chunk_minutes=10)
            
            status.info(f"3/5 Transcribing with OpenAI Whisper ({whisper_model})...")
            progress.progress(60)
            trans_segments = run_transcription(
                chunks_dir="downloads/audio_chunks",
                output_file="data/transcription.json",
                model_name=whisper_model,
                task=whisper_task
            )
            
            status.info("4/5 Indexing in ChromaDB Vector Database...")
            progress.progress(80)
            docs = create_documents(input_file="data/transcription.json", output_file="data/documents.json")
            build_vector_store(documents_input=docs)
            
            status.info("5/5 Synthesizing summary & extracting tasks...")
            progress.progress(90)
            generate_summary(
                transcription_input=trans_segments,
                output_file="data/summary.txt",
                mistral_api_key=api_key_input
            )

            tasks = extract_questions_and_tasks(
                transcription_input=trans_segments,
                mistral_api_key=api_key_input
            )
            st.session_state.extracted_tasks = tasks
            with open("data/extracted_tasks.json", "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=4)

            progress.progress(100)
            status.success("✨ Processing Complete! Video indexed successfully.")
            st.rerun()

        except Exception as e:
            status.error(f"Processing error: {str(e)}")

# =====================================================
# MAIN TABS (CLEAN & NON-CONFUSING ORGANIZED LAYOUT)
# =====================================================
tab_chat, tab_tasks, tab_summary, tab_player = st.tabs([
    "💬 AI Q&A Assistant",
    "📌 Extracted Tasks & Questions",
    "📝 Video Summary & Transcript",
    "📺 Video Player & Downloads"
])

# -----------------------------------------------------
# TAB 1: AI CHAT ASSISTANT
# -----------------------------------------------------
with tab_chat:
    if not has_vector_db:
        st.info("👋 Enter a YouTube URL or upload a video/audio file above and click 'Process Video' to start chatting.")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Video Timestamp References"):
                    for idx, src in enumerate(msg["sources"], 1):
                        start_sec = src.get("start", 0)
                        start_fmt = src.get("start_fmt", "00:00")
                        end_fmt = src.get("end_fmt", "00:00")
                        yt_link = ""
                        if st.session_state.youtube_url and "youtu" in st.session_state.youtube_url:
                            base_url = st.session_state.youtube_url.split("&")[0]
                            yt_link = f" · [▶ Play at {start_fmt}]({base_url}&t={int(start_sec)}s)"
                        st.markdown(f"**Source [{idx}]** `{start_fmt} - {end_fmt}`{yt_link}")
                        st.caption(src.get("content", ""))

    user_input = st.chat_input("Ask any question about the video content...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not has_vector_db:
                resp = "Please process a video first using the 'Process Video' button above."
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                with st.spinner("Searching video context..."):
                    try:
                        answer, sources = ask_question(
                            question=user_input,
                            chroma_dir="data/vector_db",
                            collection_name="youtube_transcript",
                            mistral_api_key=api_key_input,
                            top_k=top_k_chunks
                        )
                        st.markdown(answer)

                        if sources:
                            with st.expander("📚 View Video Timestamp References"):
                                for idx, src in enumerate(sources, 1):
                                    start_sec = src.get("start", 0)
                                    start_fmt = src.get("start_fmt", "00:00")
                                    end_fmt = src.get("end_fmt", "00:00")
                                    yt_link = ""
                                    if st.session_state.youtube_url and "youtu" in st.session_state.youtube_url:
                                        base_url = st.session_state.youtube_url.split("&")[0]
                                        yt_link = f" · [▶ Play at {start_fmt}]({base_url}&t={int(start_sec)}s)"
                                    st.markdown(f"**Source [{idx}]** `{start_fmt} - {end_fmt}`{yt_link}")
                                    st.caption(src.get("content", ""))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# -----------------------------------------------------
# TAB 2: EXTRACTED TASKS & QUESTIONS IN VIDEO
# -----------------------------------------------------
with tab_tasks:
    st.markdown('<div class="card-heading">📌 Auto-Detected Questions & Homework Asked in Video</div>', unsafe_allow_html=True)
    
    if st.session_state.extracted_tasks:
        for t_idx, item in enumerate(st.session_state.extracted_tasks, 1):
            ts = item.get("timestamp", "00:00")
            task_type = item.get("type", "Question")
            task_desc = item.get("task", "")
            explanation = item.get("explanation", "")
            
            yt_link = ""
            if st.session_state.youtube_url and "youtu" in st.session_state.youtube_url:
                parts = [int(p) for p in str(ts).split(":") if p.isdigit()]
                seconds = parts[0]*3600 + parts[1]*60 + parts[2] if len(parts)==3 else (parts[0]*60 + parts[1] if len(parts)==2 else (parts[0] if parts else 0))
                base_url = st.session_state.youtube_url.split("&")[0]
                yt_link = f" · [▶ Jump to Video at {ts}]({base_url}&t={seconds}s)"
                
            st.markdown(f"""
            <div class="info-card">
                <span class="ts-pill">{ts}</span> <strong style="color: #c084fc;">[{task_type}]</strong> {yt_link}
                <div style="font-weight: 600; color: #f8fafc; margin-top: 6px;">Q{t_idx}: {task_desc}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">💡 {explanation}</div>
            </div>
            """, unsafe_allow_html=True)
    elif has_vector_db:
        if st.button("🔍 Extract Questions & Tasks from Transcript"):
            with st.spinner("Scanning transcript..."):
                tasks = extract_questions_and_tasks(
                    transcription_input="data/transcription.json",
                    mistral_api_key=api_key_input
                )
                st.session_state.extracted_tasks = tasks
                with open("data/extracted_tasks.json", "w", encoding="utf-8") as f:
                    json.dump(tasks, f, indent=4)
                st.rerun()
    else:
        st.info("Process a video above to extract questions and homework tasks asked by the instructor.")

# -----------------------------------------------------
# TAB 3: VIDEO SUMMARY & TRANSCRIPT
# -----------------------------------------------------
with tab_summary:
    col_sum, col_trans = st.columns([1, 1])
    
    with col_sum:
        st.markdown('<div class="card-heading">📝 Executive Summary</div>', unsafe_allow_html=True)
        if has_summary:
            with open("data/summary.txt", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.caption("Summary will appear here after processing.")

    with col_trans:
        st.markdown('<div class="card-heading">📜 Searchable Full Transcript</div>', unsafe_allow_html=True)
        if has_transcription:
            with open("data/transcription.json", "r", encoding="utf-8") as f:
                trans_data = json.load(f)

            search_q = st.text_input("Filter transcript keyword:", "", placeholder="Type keyword...", label_visibility="collapsed")
            
            with st.container(height=400):
                for seg in trans_data:
                    text = seg.get("text", "")
                    if not search_q or search_q.lower() in text.lower():
                        s_fmt = format_timestamp(seg.get("start", 0))
                        e_fmt = format_timestamp(seg.get("end", 0))
                        yt_link = ""
                        if st.session_state.youtube_url and "youtu" in st.session_state.youtube_url:
                            base_url = st.session_state.youtube_url.split("&")[0]
                            ts_url = f"{base_url}&t={int(seg.get('start', 0))}s"
                            yt_link = f"[▶]({ts_url})"
                        st.markdown(f"<span class='ts-pill'>{s_fmt} - {e_fmt}</span> {yt_link} {text}", unsafe_allow_html=True)
        else:
            st.caption("No transcript available.")

# -----------------------------------------------------
# TAB 4: VIDEO PLAYER & DOWNLOADS
# -----------------------------------------------------
with tab_player:
    st.markdown('<div class="card-heading">📺 Video Player</div>', unsafe_allow_html=True)
    if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
        st.video(st.session_state.video_file_path)
    elif st.session_state.youtube_url:
        st.video(st.session_state.youtube_url)
    else:
        st.info("No video loaded.")

    st.markdown("---")
    st.markdown('<div class="card-heading">📥 Download Notes</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 Export PDF Study Guide", use_container_width=True):
            summary_str = ""
            if has_summary:
                with open("data/summary.txt", "r", encoding="utf-8") as f:
                    summary_str = f.read()
            pdf_path = generate_pdf_notes(
                video_title=st.session_state.youtube_url or "Uploaded Video Lecture",
                summary_text=summary_str,
                chat_history=st.session_state.messages
            )
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Notes", data=f, file_name="Lecture_Notes.pdf", mime="application/pdf", use_container_width=True)

    with c2:
        if has_transcription:
            with open("data/transcription.json", "r", encoding="utf-8") as f:
                t_data = json.load(f)
            sum_str = ""
            if has_summary:
                with open("data/summary.txt", "r", encoding="utf-8") as f:
                    sum_str = f.read()
            txt_data = generate_txt_export(sum_str, t_data, st.session_state.messages)
            st.download_button("📥 Download TXT Transcript", data=txt_data, file_name="Lecture_Notes.txt", mime="text/plain", use_container_width=True)
