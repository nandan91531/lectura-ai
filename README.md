# 🎓 Lectura AI --- YouTube & Lecture Study Assistant

```{=html}
<p align="center">
```
`<strong>`{=html}Transform lectures into searchable, structured, and
interactive learning material.`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
Lectura AI converts YouTube lectures and local audio files into
timestamped transcripts, AI-generated summaries, study tasks, and
context-aware answers using Retrieval-Augmented Generation (RAG).
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<a href="https://www.python.org/">`{=html}
`<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">`{=html}
`</a>`{=html} `<a href="https://fastapi.tiangolo.com/">`{=html}
`<img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">`{=html}
`</a>`{=html} `<a href="https://streamlit.io/">`{=html}
`<img src="https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">`{=html}
`</a>`{=html} `<a href="https://www.langchain.com/">`{=html}
`<img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat" alt="LangChain">`{=html}
`</a>`{=html} `<a href="https://github.com/openai/whisper">`{=html}
`<img src="https://img.shields.io/badge/Whisper-Speech--to--Text-412991?style=flat&logo=openai&logoColor=white" alt="Whisper">`{=html}
`</a>`{=html} `<a href="https://www.trychroma.com/">`{=html}
`<img src="https://img.shields.io/badge/Vector%20Database-ChromaDB-orange?style=flat" alt="ChromaDB">`{=html}
`</a>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

**Lectura AI** is an AI-powered lecture learning assistant that converts
long educational videos and audio recordings into a structured,
searchable knowledge base.

The application supports two input methods:

-   YouTube lecture URLs
-   Locally uploaded audio files such as `.mp3`, `.wav`, and `.m4a`

The processing pipeline downloads or stores the audio, converts it into
a standard format, divides it into manageable chunks, transcribes it
using **OpenAI Whisper**, and converts the resulting transcript into
documents suitable for semantic search.

The application then uses **HuggingFace sentence-transformer
embeddings** and **ChromaDB** to support Retrieval-Augmented Generation
(RAG). Users can ask questions about the lecture, receive context-aware
answers, review summaries and study tasks, and export their notes in PDF
or TXT format.

> **Implementation note:** Features and technical details in this README
> should be kept synchronized with the actual source code. Any feature
> marked as planned or proposed must not be presented as implemented.

------------------------------------------------------------------------

## ✨ Key Features

### 🎙️ Multi-Source Audio Ingestion

-   Accept a YouTube URL or a local audio upload.
-   Download YouTube audio using `yt-dlp`.
-   Convert audio into WAV format using audio-processing utilities and
    FFmpeg.
-   Split long recordings into smaller chunks for manageable processing.
-   Store intermediate files in the runtime `downloads/` directory.

### ⚡ Local Speech-to-Text

-   Use OpenAI Whisper for speech recognition.
-   Process audio chunks individually.
-   Preserve transcription segments and timestamps.
-   Store transcription output in JSON format.
-   Support configurable Whisper model names and transcription tasks.

### 🧠 Retrieval-Augmented Generation

-   Convert transcript content into structured documents.
-   Generate vector embeddings using HuggingFace sentence-transformer
    models.
-   Store document vectors in ChromaDB.
-   Retrieve relevant lecture segments for user questions.
-   Generate answers using the configured Mistral AI integration.
-   Return relevant source information when supported by the
    implementation.

### 🤖 AI Summaries and Study Tasks

-   Generate a lecture summary using Mistral AI.
-   Extract important questions, concepts, and study tasks.
-   Save generated outputs for later access.
-   Allow users to review lecture content without replaying the entire
    recording.

### 💬 Interactive Q&A

-   Ask questions about the indexed lecture.
-   Retrieve the most relevant transcript chunks.
-   Construct context for the language model.
-   Generate a response based on retrieved lecture content.
-   Maintain chat history during the application session where
    supported.

### 📄 Export

-   Export lecture notes as PDF.
-   Export transcript, summary, and chat information as TXT.
-   Store generated export files in the `downloads/` directory.

### 🖥️ Dual Interface

The repository describes two interfaces:

1.  **Streamlit interface**
    -   Implemented through `app.py`.
    -   Intended for an interactive dashboard and cloud deployment.
2.  **FastAPI interface**
    -   Implemented through `server.py`.
    -   Provides REST endpoints and serves the custom frontend in
        `static/index.html`.

------------------------------------------------------------------------

## 🏗️ System Architecture

``` mermaid
flowchart TD
    U[User] --> UI1[Streamlit UI]
    U --> UI2[HTML/CSS/JavaScript UI]

    UI1 --> API[Application Logic]
    UI2 --> FASTAPI[FastAPI Server]
    FASTAPI --> API

    API --> INPUT{Input Type}
    INPUT -->|YouTube URL| YT[yt-dlp Audio Download]
    INPUT -->|Local Audio| UPLOAD[Save Uploaded File]

    YT --> AUDIO[Audio Processing]
    UPLOAD --> AUDIO

    AUDIO --> WAV[Convert to WAV]
    WAV --> CHUNKS[Split Audio into Chunks]
    CHUNKS --> WHISPER[OpenAI Whisper]
    WHISPER --> TRANSCRIPT[Timestamped JSON Transcript]

    TRANSCRIPT --> DOCS[Document Creation and Chunking]
    DOCS --> EMBED[HuggingFace Embeddings]
    EMBED --> VECTOR[(ChromaDB Vector Store)]

    TRANSCRIPT --> SUMMARY[Mistral AI Summary]
    TRANSCRIPT --> TASKS[Question and Task Extraction]

    USERQ[User Question] --> RETRIEVE[Retrieve Relevant Chunks]
    VECTOR --> RETRIEVE
    RETRIEVE --> CONTEXT[Build Context]
    CONTEXT --> LLM[Mistral AI]
    LLM --> ANSWER[Answer with Sources/Timestamps]

    SUMMARY --> EXPORT[PDF/TXT Export]
    TASKS --> EXPORT
    ANSWER --> EXPORT
    TRANSCRIPT --> EXPORT
```

### Main Processing Stages

1.  **Input:** YouTube URL or uploaded audio file.
2.  **Audio processing:** Download, conversion, and chunking.
3.  **Transcription:** Convert audio into timestamped text.
4.  **Document preparation:** Convert transcript segments into
    searchable documents.
5.  **Embedding:** Transform text into numerical vectors.
6.  **Indexing:** Store vectors and metadata in ChromaDB.
7.  **AI analysis:** Generate summaries and study tasks.
8.  **RAG Q&A:** Retrieve relevant context and generate answers.
9.  **Export:** Produce PDF or TXT lecture notes.

------------------------------------------------------------------------

## 🔄 End-to-End Project Flow

``` mermaid
flowchart TD
    START([Start]) --> INPUT[Enter YouTube URL or Upload Audio]
    INPUT --> VALIDATE{Is Input Valid?}

    VALIDATE -->|No| ERROR1[Return Validation Error]
    VALIDATE -->|Yes| SOURCE{Input Source}

    SOURCE -->|YouTube| DOWNLOAD[Download Audio with yt-dlp]
    SOURCE -->|Upload| SAVE[Save Uploaded Audio]

    DOWNLOAD --> CONVERT[Convert Audio to WAV]
    SAVE --> CONVERT

    CONVERT --> SPLIT[Split Audio into Chunks]
    SPLIT --> TRANSCRIBE[Transcribe Chunks with Whisper]
    TRANSCRIBE --> JSON[Save Transcription JSON]

    JSON --> DOCUMENTS[Create Documents]
    DOCUMENTS --> EMBEDDINGS[Generate Embeddings]
    EMBEDDINGS --> CHROMA[Build ChromaDB Vector Store]

    JSON --> SUMMARIZE[Generate Summary]
    JSON --> EXTRACT[Extract Questions and Tasks]

    CHROMA --> READY[Lecture Knowledge Base Ready]
    SUMMARIZE --> READY
    EXTRACT --> READY

    READY --> USER_ACTION{User Action}
    USER_ACTION -->|Read Transcript| TRANSCRIPT_VIEW[Show Transcript]
    USER_ACTION -->|Read Summary| SUMMARY_VIEW[Show Summary]
    USER_ACTION -->|View Tasks| TASK_VIEW[Show Tasks]
    USER_ACTION -->|Ask Question| RAG[Run RAG Pipeline]
    USER_ACTION -->|Export| EXPORT[Generate PDF or TXT]

    RAG --> RESPONSE[Return Answer and Sources]
    TRANSCRIPT_VIEW --> END([End])
    SUMMARY_VIEW --> END
    TASK_VIEW --> END
    RESPONSE --> END
    EXPORT --> END
    ERROR1 --> END
```

------------------------------------------------------------------------

## 📁 Repository Structure

``` text
rag-based-ai/
│
├── app.py                     # Streamlit application
├── server.py                  # FastAPI server and REST endpoints
├── requirements.txt           # Python dependencies
├── packages.txt               # System dependencies for supported hosting
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Docker Compose configuration
├── .env                       # Local environment variables (do not commit)
├── .gitignore                 # Files excluded from Git
│
├── core/
│   ├── transcriber.py         # Whisper transcription pipeline
│   ├── document_creator.py    # Transcript-to-document conversion
│   ├── vector_store.py        # ChromaDB vector index creation
│   ├── rag_engine.py          # Retrieval and answer generation
│   ├── summarizer.py          # AI summary generation
│   └── task_extractor.py      # Question/task extraction
│
├── utils/
│   ├── audio_processor.py     # Download, conversion, and chunking
│   └── exporter.py             # PDF and TXT generation
│
├── static/
│   └── index.html              # FastAPI web interface
│
├── data/                       # Generated transcript and vector data
└── downloads/                  # Audio, temporary files, and exports
```

------------------------------------------------------------------------

## 🧩 Core Modules

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  `audio_processor.py`                Downloads audio, converts formats,
                                      and splits audio

  `transcriber.py`                    Runs Whisper transcription and
                                      generates transcript output

  `document_creator.py`               Converts transcript data into
                                      structured documents

  `vector_store.py`                   Creates and persists the ChromaDB
                                      vector index

  `rag_engine.py`                     Retrieves relevant documents and
                                      generates answers

  `summarizer.py`                     Generates AI-based lecture
                                      summaries

  `task_extractor.py`                 Extracts questions, tasks, or study
                                      actions

  `exporter.py`                       Creates PDF and TXT exports

  `server.py`                         Exposes API endpoints and
                                      coordinates the pipeline

  `app.py`                            Provides the Streamlit interface
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Technology Stack

  Technology               Purpose
  ------------------------ -------------------------------------------------
  Python                   Main programming language
  FastAPI                  REST API backend
  Streamlit                Interactive web interface
  OpenAI Whisper           Speech-to-text transcription
  yt-dlp                   YouTube audio extraction
  FFmpeg                   Audio conversion and processing
  pydub                    Audio manipulation and chunking
  LangChain                Document, retrieval, and LLM orchestration
  HuggingFace Embeddings   Text-to-vector conversion
  Sentence Transformers    Embedding model implementation
  ChromaDB                 Vector database
  Mistral AI               Summary, task extraction, and answer generation
  Pydantic                 Request and data validation
  fpdf2                    PDF generation
  Docker                   Containerized deployment
  python-dotenv            Environment variable management

------------------------------------------------------------------------

## 🚀 Installation

### Prerequisites

Install the following software:

-   Python 3.10 or later
-   FFmpeg
-   Git
-   Optional: Docker and Docker Compose

### Install FFmpeg

#### Windows

``` powershell
winget install ffmpeg
```

Alternative:

``` powershell
choco install ffmpeg
```

#### macOS

``` bash
brew install ffmpeg
```

#### Ubuntu/Debian

``` bash
sudo apt update
sudo apt install ffmpeg
```

Verify the installation:

``` bash
ffmpeg -version
```

### Clone the Repository

``` bash
git clone https://github.com/nandan91531/lectura-ai.git
cd lectura-ai
```

### Create a Virtual Environment

#### Windows PowerShell

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS/Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔐 Environment Configuration

Create a `.env` file in the project root:

``` env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Never commit the `.env` file to GitHub.

Add the following to `.gitignore`:

``` gitignore
.env
.venv/
__pycache__/
*.pyc
data/
downloads/
```

The exact environment variables required by the project should be
verified against the implementation before deployment.

------------------------------------------------------------------------

## ▶️ Running the Application

### Option A: Run the FastAPI Server

``` bash
python server.py
```

Open:

``` text
http://localhost:8000
```

### Option B: Run FastAPI with Uvicorn

``` bash
uvicorn server:app --reload --port 8000
```

FastAPI documentation:

``` text
http://localhost:8000/docs
```

Alternative ReDoc documentation:

``` text
http://localhost:8000/redoc
```

### Option C: Run the Streamlit Application

``` bash
streamlit run app.py
```

The terminal will display the local Streamlit URL.

### Option D: Run with Docker Compose

``` bash
docker compose up -d --build
```

To view logs:

``` bash
docker compose logs -f
```

To stop the services:

``` bash
docker compose down
```

------------------------------------------------------------------------

## 🔌 FastAPI Endpoints

The following endpoints are described by the project documentation and
should be validated against the current `server.py` implementation.

  ------------------------------------------------------------------------
  Method                  Endpoint                 Description
  ----------------------- ------------------------ -----------------------
  `GET`                   `/`                      Serves the web
                                                   interface

  `GET`                   `/api/status`            Returns processing and
                                                   storage status

  `POST`                  `/api/ingest`            Starts lecture
                                                   ingestion

  `POST`                  `/api/chat`              Answers a question
                                                   using RAG

  `GET`                   `/api/summary`           Retrieves the generated
                                                   summary

  `GET`                   `/api/transcript`        Retrieves the
                                                   transcription

  `GET`                   `/api/tasks`             Retrieves extracted
                                                   tasks/questions

  `POST`                  `/api/tasks/reextract`   Re-extracts tasks from
                                                   transcription

  `GET`                   `/api/export/pdf`        Generates or downloads
                                                   PDF notes

  `GET`                   `/api/export/txt`        Generates or downloads
                                                   TXT notes
  ------------------------------------------------------------------------

### Example: Start Ingestion

The exact multipart field names should be checked in the API schema or
source code. Conceptually, ingestion accepts:

-   YouTube URL or uploaded audio
-   Whisper model name
-   Transcription task
-   Mistral API key where required

### Example: Ask a Question

``` json
{
  "question": "Explain the main concept discussed in the lecture.",
  "top_k": 2,
  "api_key": "your_key_if_required"
}
```

Do not place real API keys in public documentation, screenshots, GitHub
issues, or client-side code.

------------------------------------------------------------------------

## 🧠 RAG Pipeline

Lectura AI uses a Retrieval-Augmented Generation workflow rather than
relying only on the language model's general knowledge.

``` mermaid
flowchart TD
    A[Timestamped Transcript] --> B[Create Documents]
    B --> C[Split or Prepare Text Chunks]
    C --> D[Generate Embeddings]
    D --> E[(ChromaDB)]

    Q[User Question] --> F[Embed Query]
    F --> G[Similarity Search]
    E --> G
    G --> H[Retrieve Top-k Documents]
    H --> I[Build Context]
    I --> J[Send Context and Question to LLM]
    J --> K[Generate Answer]
    K --> L[Return Answer and Source Information]
```

### Why RAG Is Used

A language model does not automatically know the content of a private
lecture recording. RAG addresses this by:

1.  Converting lecture content into searchable documents.
2.  Creating embeddings for the documents.
3.  Retrieving the most relevant content for a question.
4.  Supplying the retrieved content to the language model.
5.  Generating an answer grounded in the retrieved context.

### RAG Components

  Component         Function
  ----------------- ------------------------------------------
  Documents         Structured transcript content
  Embeddings        Numerical representation of text meaning
  Vector database   Stores embeddings and metadata
  Retriever         Finds relevant transcript chunks
  Context builder   Combines retrieved content
  LLM               Generates the final response

------------------------------------------------------------------------

## 🎙️ Audio and Transcription Pipeline

``` mermaid
flowchart TD
    A[YouTube URL or Audio Upload] --> B{Source}
    B -->|YouTube| C[yt-dlp Download]
    B -->|Local File| D[Save Uploaded File]
    C --> E[Audio File]
    D --> E
    E --> F[Convert to WAV]
    F --> G[Split into Audio Chunks]
    G --> H[Load Whisper Model]
    H --> I[Transcribe Each Chunk]
    I --> J[Collect Text and Timestamps]
    J --> K[Save transcription.json]
```

### Important Considerations

-   Long recordings may require substantial CPU, RAM, and storage.
-   Whisper processing time depends on the model size and available
    hardware.
-   Audio quality can affect transcription quality.
-   Chunk boundaries can affect context and timestamp continuity.
-   Temporary files should be cleaned up after processing where
    appropriate.

------------------------------------------------------------------------

## 📊 Data and Storage

The project uses a combination of generated files and vector storage.

  Data                     Purpose
  ------------------------ -----------------------------------------
  Downloaded audio         Original or downloaded input
  WAV files                Standardized audio for processing
  Audio chunks             Smaller units for transcription
  `transcription.json`     Timestamped transcript
  `documents.json`         Structured document representation
  Vector database          Embeddings and searchable metadata
  `summary.txt`            Generated lecture summary
  `extracted_tasks.json`   Extracted questions and study tasks
  Chat history             Conversation context during the session
  PDF/TXT exports          Downloadable lecture notes

The exact persistence behavior, cleanup strategy, and multi-user
isolation should be confirmed from the implementation.

------------------------------------------------------------------------

## 🧪 Testing

Recommended testing areas include:

### Unit Testing

Test individual functions such as:

-   Audio conversion
-   Audio splitting
-   Transcript parsing
-   Document creation
-   Embedding creation
-   Export generation
-   Timestamp formatting

### Integration Testing

Test complete workflows such as:

-   YouTube URL to transcript
-   Uploaded audio to vector database
-   User question to RAG answer
-   Summary generation to export
-   Task extraction from transcript

### API Testing

Validate:

-   Request schema
-   Missing fields
-   Invalid URLs
-   Unsupported file types
-   Missing vector database
-   Missing API keys
-   Concurrent ingestion requests
-   Export failures

### Example Test Checklist

  Test Case              Expected Behavior
  ---------------------- -------------------------------------
  Valid YouTube URL      Audio ingestion starts
  Valid audio upload     File is saved and processed
  Missing API key        Request is rejected where required
  Invalid input          Validation error is returned
  Chat before indexing   Informative error is returned
  Missing vector store   Chat request is rejected
  Export without data    Safe error or empty response
  Repeated ingestion     Concurrent processing is controlled

Actual test results should only be documented after running the tests
and recording their output.

------------------------------------------------------------------------

## 🔒 Security Considerations

The following controls are recommended for a production deployment:

-   Store API keys in environment variables or a secret manager.
-   Never expose API keys in frontend JavaScript.
-   Validate uploaded file extensions and MIME types.
-   Restrict upload size.
-   Prevent path traversal through sanitized filenames.
-   Configure CORS for trusted origins instead of allowing every origin.
-   Add authentication and authorization for multi-user deployments.
-   Add rate limiting to expensive AI endpoints.
-   Avoid exposing internal exception details to users.
-   Isolate user-specific data and vector collections.
-   Clean up temporary files.
-   Add logging without recording secrets.

> Security recommendations are not proof that these controls are already
> implemented. Verify each control against the current source code.

------------------------------------------------------------------------

## 🐳 Deployment

### Local Deployment

The recommended local workflow is:

``` mermaid
flowchart TD
    A[Developer] --> B[Create Virtual Environment]
    B --> C[Install Python Dependencies]
    C --> D[Install FFmpeg]
    D --> E[Configure Environment Variables]
    E --> F[Run FastAPI or Streamlit]
    F --> G[Test Application Locally]
```

### Docker Deployment

``` mermaid
flowchart TD
    A[Source Code] --> B[Dockerfile]
    B --> C[Install Dependencies]
    C --> D[Configure Runtime]
    D --> E[Expose Application Port]
    E --> F[Run Container]
    F --> G[Access Application]
```

Before production deployment, verify:

-   FFmpeg availability inside the container
-   Whisper model download behavior
-   Persistent storage configuration
-   Environment variable injection
-   Port configuration
-   CPU and memory requirements
-   Request timeouts
-   Background job reliability
-   Logging and monitoring

------------------------------------------------------------------------

## ☁️ Streamlit Community Cloud

A possible deployment workflow for the Streamlit interface is:

1.  Push the project to GitHub.
2.  Open [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Create a new application.
4.  Select the repository and `app.py`.
5.  Configure required secrets.
6.  Deploy and review application logs.

Example secret configuration:

``` toml
MISTRAL_API_KEY = "your_mistral_api_key_here"
```

Deployment compatibility must be verified because speech recognition,
FFmpeg, vector storage, and model inference may require significant
resources.

------------------------------------------------------------------------

## ⚖️ Design Trade-Offs

  -----------------------------------------------------------------------
  Decision                Benefit                 Limitation
  ----------------------- ----------------------- -----------------------
  Whisper                 Local transcription and Can be
                          broad language support  resource-intensive

  ChromaDB                Simple vector storage   Requires persistence
                          and retrieval           and lifecycle planning

  HuggingFace embeddings  Supports semantic       Model choice affects
                          search                  quality and resource
                                                  usage

  Mistral API             Reduces need for local  Requires API access,
                          LLM inference           cost control, and key
                                                  protection

  FastAPI                 API validation and      Production scaling
                          interactive             needs additional
                          documentation           infrastructure

  Background processing   Prevents long ingestion Needs job tracking and
                          work from blocking the  reliable failure
                          request                 recovery

  File-based storage      Simple local            Less suitable for
                          development             multi-user and
                                                  distributed systems

  Streamlit               Fast interface          Complex frontend state
                          development             and large-scale
                                                  workloads may require
                                                  additional architecture
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🚧 Known Limitations to Review

The following areas should be evaluated before presenting the project as
production-ready:

-   Long-running transcription jobs may require a dedicated job queue.
-   In-memory state may not survive server restarts.
-   A shared vector database may not provide user-level isolation.
-   Open CORS configuration can create security risks.
-   Large audio files can increase memory, CPU, and disk usage.
-   Timestamp accuracy may be affected by chunking and
    transcript-merging logic.
-   API failures and rate limits require robust retry and fallback
    behavior.
-   Cloud hosting may impose limits on processing time, memory, storage,
    and model downloads.
-   Chat history persistence may be limited if it is stored only in
    application memory.

These are review areas and should be confirmed against the actual
codebase.

------------------------------------------------------------------------

## 🔮 Future Enhancements

Potential improvements include:

### Retrieval Improvements

-   Timestamp-aware semantic chunking
-   Overlapping chunks for better context continuity
-   Hybrid keyword and vector search
-   Reranking retrieved documents
-   Improved source citation and timestamp linking
-   Query rewriting for ambiguous questions

### Processing Improvements

-   Background job queues using Celery or another task system
-   Redis-based task status tracking
-   Parallel audio-chunk transcription
-   Automatic temporary-file cleanup
-   Progress reporting
-   Retry and checkpoint support

### Product Improvements

-   User authentication
-   Multiple lecture workspaces
-   Saved lecture collections
-   Search across multiple lectures
-   Flashcard generation
-   Quiz generation
-   Spaced repetition
-   Multilingual translation
-   Speaker identification
-   Lecture chapter detection

### Production Improvements

-   Structured logging
-   Monitoring and health checks
-   Rate limiting
-   Persistent relational metadata storage
-   Object storage for large audio files
-   CI/CD automation
-   Automated test coverage
-   Secure secret management

------------------------------------------------------------------------

## 🗺️ Suggested Development Roadmap

``` mermaid
flowchart LR
    A[Core Ingestion] --> B[Transcription]
    B --> C[Document and Embedding Pipeline]
    C --> D[Vector Search]
    D --> E[RAG Q&A]
    E --> F[Summary and Task Extraction]
    F --> G[Export Features]
    G --> H[Testing and Security]
    H --> I[Deployment and Monitoring]
```

### Phase 1 --- Core Pipeline

-   Validate YouTube and upload ingestion.
-   Stabilize audio conversion and chunking.
-   Generate reliable timestamped transcription.

### Phase 2 --- Knowledge Base

-   Improve document creation.
-   Verify embedding configuration.
-   Build and test ChromaDB persistence.

### Phase 3 --- RAG

-   Validate retrieval quality.
-   Improve prompt construction.
-   Test answers against known lecture content.
-   Verify source and timestamp handling.

### Phase 4 --- User Experience

-   Improve loading and error states.
-   Add clear processing progress.
-   Improve transcript, summary, and task views.
-   Validate PDF and TXT exports.

### Phase 5 --- Production Readiness

-   Add authentication if required.
-   Secure API keys and CORS.
-   Add logging, monitoring, and rate limiting.
-   Introduce a reliable background job system.
-   Add automated tests and deployment checks.

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome.

### Suggested Workflow

``` bash
git checkout -b feature/your-feature-name
```

Make your changes, test them locally, and commit:

``` bash
git add .
git commit -m "Add: describe your change"
git push origin feature/your-feature-name
```

Then open a pull request with:

-   Problem description
-   Proposed solution
-   Files changed
-   Testing performed
-   Known limitations

Do not commit:

-   `.env` files
-   API keys
-   Private recordings
-   Generated runtime data
-   Large model files
-   Personal information

------------------------------------------------------------------------

## 🧾 License

This project is described as being distributed under the **MIT
License**.

Make sure a valid `LICENSE` file exists in the repository before
publishing the project with this license.

------------------------------------------------------------------------

## 👨‍💻 Author

**Your Name**

-   GitHub: [nandan91531](https://github.com/nandan91531)
-   LinkedIn: `https://www.linkedin.com/in/YOUR_PROFILE`

Review the repository name and update any remaining project-specific placeholders before publishing.

------------------------------------------------------------------------

## ❤️ Acknowledgements

This project uses and builds upon open-source technologies including:

-   [OpenAI Whisper](https://github.com/openai/whisper)
-   [FastAPI](https://fastapi.tiangolo.com/)
-   [Streamlit](https://streamlit.io/)
-   [LangChain](https://www.langchain.com/)
-   [ChromaDB](https://www.trychroma.com/)
-   [Sentence Transformers](https://www.sbert.net/)
-   [yt-dlp](https://github.com/yt-dlp/yt-dlp)
-   [FFmpeg](https://ffmpeg.org/)

------------------------------------------------------------------------

```{=html}
<p align="center">
```
Built for students, researchers, and lifelong learners.
```{=html}
</p>
```
