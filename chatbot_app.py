"""
SmartRAG ChatBot - Multimodal Study Assistant

Features:
- PDF / DOCX / TXT / Markdown ingestion
- Image and audio ingestion
- Local RAG Q&A with Ollama + Qwen
- ChromaDB semantic retrieval
- Grounded study summaries
- Grounded quiz generation
- SQLite file storage
- PDF / image / audio / text viewer
- CPU-only Whisper to avoid GPU VRAM conflicts
"""

import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from multimodal_rag.base import QueryRequest
from multimodal_rag.system import MultimodalRAGSystem


# ============================================================
# OPTIONAL CONFIGURATION
# ============================================================
try:
    from config_schema import load_config
    USE_NEW_CONFIG = True
except ImportError:
    USE_NEW_CONFIG = False


# ============================================================
# OPTIONAL WHISPER
# ============================================================
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    WHISPER_AVAILABLE = False


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="SmartRAG Study Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6 !important;
        color: #333333 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] * {
        color: #333333 !important;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }

    .stMarkdown {
        color: #333333 !important;
    }

    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stButton > button:hover {
        background-color: #357abd;
    }

    button[kind="secondary"] {
        background-color: #e8e8e8 !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
    }

    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #cccccc;
        border-radius: 25px;
        padding: 12px 20px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
    }

    .stFileUploader {
        background-color: #f5f5f5 !important;
        border: 2px dashed #4a90e2 !important;
        border-radius: 8px;
    }

    .stFileUploader * {
        color: #333333 !important;
    }

    .stProgress > div > div > div {
        background-color: #4a90e2;
    }

    .user-message {
        background: #dcf8c6 !important;
        color: #333333 !important;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .assistant-message {
        background: #e9f5ff !important;
        color: #333333 !important;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #4a90e2;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .study-summary,
    .study-tools {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin: 12px 0 20px 0;
    }

    .study-summary {
        border-left: 5px solid #4a90e2;
    }

    .quiz-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin: 12px 0 20px 0;
    }

    .stMetric,
    .stMetric *,
    [data-testid="metric-container"],
    [data-testid="metric-container"] * {
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SQLITE DATABASE
# ============================================================
def init_file_storage_db() -> None:
    """Create the SQLite file storage database."""
    with sqlite3.connect("file_storage.db") as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stored_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_content BLOB NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT UNIQUE NOT NULL
            )
            """
        )
        conn.commit()


def store_file_in_db(
    filename: str,
    file_content: bytes,
    file_type: str,
    file_size: int,
    file_hash: str,
    upload_time: Optional[str] = None,
) -> bool:
    """Store an uploaded file in SQLite."""
    try:
        upload_time = upload_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect("file_storage.db") as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO stored_files
                (filename, file_type, file_size, file_content, file_hash, upload_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    file_type,
                    file_size,
                    file_content,
                    file_hash,
                    upload_time,
                ),
            )
            conn.commit()
        return True
    except Exception as exc:
        st.error(f"Error storing file in database: {exc}")
        return False


def get_file_from_db(filename: str):
    """Retrieve the most recent file with a given filename."""
    try:
        with sqlite3.connect("file_storage.db") as conn:
            cursor = conn.execute(
                """
                SELECT filename, file_type, file_size, file_content, upload_time
                FROM stored_files
                WHERE filename = ?
                ORDER BY upload_time DESC
                LIMIT 1
                """,
                (filename,),
            )
            return cursor.fetchone()
    except Exception as exc:
        st.error(f"Error retrieving file: {exc}")
        return None


def get_all_stored_files() -> list:
    """Return all stored file metadata."""
    try:
        with sqlite3.connect("file_storage.db") as conn:
            cursor = conn.execute(
                """
                SELECT filename, file_type, file_size, upload_time
                FROM stored_files
                ORDER BY upload_time DESC
                """
            )
            return cursor.fetchall()
    except Exception as exc:
        st.error(f"Error getting stored files: {exc}")
        return []


def delete_file_from_db(filename: str) -> bool:
    """Delete all database entries for a filename."""
    try:
        with sqlite3.connect("file_storage.db") as conn:
            conn.execute(
                "DELETE FROM stored_files WHERE filename = ?",
                (filename,),
            )
            conn.commit()
        return True
    except Exception as exc:
        st.error(f"Delete failed: {exc}")
        return False


def clear_database_files() -> None:
    """Delete all stored files."""
    try:
        with sqlite3.connect("file_storage.db") as conn:
            conn.execute("DELETE FROM stored_files")
            conn.commit()
        st.session_state.uploaded_files = []
        save_uploaded_files_list()
        st.success("✅ All stored files cleared.")
    except Exception as exc:
        st.error(f"Error clearing database: {exc}")


# ============================================================
# FILE TIMESTAMP SEARCH
# ============================================================
def search_files_by_timestamp(query_time: str) -> str:
    """Find files uploaded within ±30 minutes of a requested time."""
    match = re.search(r"(\d{1,2})[:.](\d{2})", query_time)
    if not match:
        return (
            "Could not parse the time. "
            "Use a format such as 14:20 or 14.20."
        )

    hour = int(match.group(1))
    minute = int(match.group(2))

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return "Please provide a valid time."

    today = datetime.now().date()
    target_time = datetime.combine(
        today,
        datetime.min.time().replace(hour=hour, minute=minute),
    )
    start_time = target_time - timedelta(minutes=30)
    end_time = target_time + timedelta(minutes=30)

    try:
        with sqlite3.connect("file_storage.db") as conn:
            cursor = conn.execute(
                """
                SELECT filename, file_type, upload_time
                FROM stored_files
                WHERE upload_time BETWEEN ? AND ?
                ORDER BY upload_time DESC
                """,
                (
                    start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            results = cursor.fetchall()
    except Exception as exc:
        return f"Error searching upload history: {exc}"

    if not results:
        return f"No documents were uploaded around {query_time}."

    if len(results) == 1:
        filename, file_type, upload_time = results[0]
        return (
            f"At {query_time}, you uploaded **{filename}** "
            f"({file_type}) at {upload_time}."
        )

    lines = [f"Around {query_time}, you uploaded:", ""]
    for filename, _, upload_time in results:
        lines.append(f"- **{filename}** at {upload_time}")
    return "\n".join(lines)


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "rag_system": None,
        "uploaded_files": [],
        "system_initialized": False,
        "whisper_model": None,
        "transcribed_message": "",
        "input_hash": "",
        "viewing_file": None,
        "show_file_viewer": False,
        "audio_processing": False,
        "show_audio_upload": False,
        "hide_recent_uploads": False,
        "show_emergency_sidebar": False,
        "study_summary": "",
        "study_quiz": "",
        "summary_topic": "all",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    init_file_storage_db()


# ============================================================
# WHISPER - CPU ONLY
# ============================================================
def get_whisper_model():
    """Load Whisper lazily on CPU to preserve GPU VRAM for Ollama."""
    if not WHISPER_AVAILABLE:
        return None
    if st.session_state.whisper_model is not None:
        return st.session_state.whisper_model

    with st.spinner("🎙️ Loading Whisper model on CPU..."):
        try:
            st.session_state.whisper_model = whisper.load_model(
                "base",
                device="cpu",
            )
            st.success("✅ Whisper loaded on CPU.")
        except Exception as exc:
            st.error(f"❌ Failed to load Whisper: {exc}")
            st.session_state.whisper_model = None

    return st.session_state.whisper_model


def transcribe_audio_file(audio_file) -> Optional[str]:
    """Transcribe uploaded audio using CPU Whisper."""
    if not WHISPER_AVAILABLE:
        st.error(
            "Whisper is not installed. "
            "Install it with: pip install openai-whisper"
        )
        return None

    model = get_whisper_model()
    if model is None:
        return None

    temp_path = None
    try:
        suffix = Path(audio_file.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_file.getbuffer())
            temp_path = temp_file.name

        st.session_state.audio_processing = True

        with st.spinner("🎙️ Transcribing audio..."):
            result = model.transcribe(temp_path, fp16=False)

        transcript = result.get("text", "").strip()

        if not transcript:
            st.warning("No speech detected.")
            return None

        st.success("✅ Audio transcription completed.")
        return transcript

    except Exception as exc:
        st.error(f"Transcription error: {exc}")
        return None

    finally:
        st.session_state.audio_processing = False
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ============================================================
# FILE VIEWER
# ============================================================
def render_file_viewer() -> None:
    """Display a stored file."""
    filename = st.session_state.viewing_file
    if not filename:
        return

    file_data = get_file_from_db(filename)
    if not file_data:
        st.error(f"File '{filename}' was not found.")
        if st.button("Close", key="viewer_missing_close"):
            st.session_state.viewing_file = None
            st.session_state.show_file_viewer = False
            st.rerun()
        return

    (
        filename,
        file_type,
        file_size,
        file_content,
        upload_time,
    ) = file_data

    st.markdown(f"## 📄 {filename}")
    st.caption(
        f"Type: {file_type} | Size: {file_size:,} bytes | Uploaded: {upload_time}"
    )

    if st.button("❌ Close Viewer", key="close_viewer"):
        st.session_state.viewing_file = None
        st.session_state.show_file_viewer = False
        st.rerun()

    st.markdown("---")

    if file_type.startswith("image/"):
        try:
            image = Image.open(BytesIO(file_content))
            st.image(image, caption=filename, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not display image: {exc}")

    elif file_type == "application/pdf":
        encoded_pdf = base64.b64encode(file_content).decode("utf-8")
        pdf_html = f"""
        <iframe
            src="data:application/pdf;base64,{encoded_pdf}"
            width="100%"
            height="700"
            style="border:2px solid #4a90e2; border-radius:10px;">
        </iframe>
        """
        st.markdown(pdf_html, unsafe_allow_html=True)
        st.download_button(
            "📥 Download PDF",
            data=file_content,
            file_name=filename,
            mime=file_type,
            use_container_width=True,
        )

    elif file_type.startswith("audio/"):
        st.audio(file_content, format=file_type)
        st.download_button(
            "📥 Download Audio",
            data=file_content,
            file_name=filename,
            mime=file_type,
        )

    elif file_type.startswith("text/"):
        try:
            text = file_content.decode("utf-8")
            st.text_area("File Content", text, height=500)
        except Exception:
            st.download_button(
                "📥 Download File",
                data=file_content,
                file_name=filename,
                mime=file_type,
            )

    else:
        st.download_button(
            "📥 Download File",
            data=file_content,
            file_name=filename,
            mime=file_type,
        )


# ============================================================
# RAG SYSTEM
# ============================================================
def get_rag_system():
    """Initialize and cache SmartRAG."""
    if st.session_state.rag_system is not None:
        return st.session_state.rag_system

    with st.spinner("🔧 Initializing SmartRAG..."):
        try:
            if USE_NEW_CONFIG:
                load_config(config_path="config.yaml")

            system = MultimodalRAGSystem(
                config_path="config.yaml"
            )

            if not system.is_available():
                st.error(
                    "❌ SmartRAG initialized, but the local LLM is unavailable."
                )
                st.session_state.system_initialized = False
                return None

            st.session_state.rag_system = system
            st.session_state.system_initialized = True

            st.success("✅ SmartRAG initialized successfully!")
            return system

        except Exception as exc:
            st.error(f"❌ SmartRAG initialization failed: {exc}")
            st.session_state.rag_system = None
            st.session_state.system_initialized = False
            return None


# ============================================================
# UPLOADED FILE LIST
# ============================================================
def save_uploaded_files_list() -> None:
    """Save uploaded file metadata."""
    try:
        os.makedirs("user_data", exist_ok=True)
        with open(
            "user_data/streamlit_uploaded_files.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                st.session_state.uploaded_files,
                file,
                indent=2,
                default=str,
            )
    except Exception as exc:
        st.error(f"Error saving uploaded-file list: {exc}")


def load_uploaded_files_list() -> None:
    """Load uploaded file metadata."""
    path = Path("user_data/streamlit_uploaded_files.json")
    if not path.exists():
        return

    try:
        with open(path, "r", encoding="utf-8") as file:
            st.session_state.uploaded_files = json.load(file)
    except Exception as exc:
        st.error(f"Error loading uploaded-file list: {exc}")


# ============================================================
# FILE PROCESSING
# ============================================================
def process_uploaded_file(uploaded_file, rag_system):
    """Process one file with SmartRAG."""
    temp_path = None

    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        safe_name = Path(uploaded_file.name).name
        temp_path = temp_dir / safe_name

        uploaded_file.seek(0)
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        start_time = time.time()
        result = rag_system.ingest_file(temp_path)
        elapsed = time.time() - start_time

        if not result.success:
            return (
                False,
                f"❌ Failed to process {uploaded_file.name}: {result.error_message}",
            )

        file_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": len(result.chunks),
            "processing_time": (
                result.processing_time
                if result.processing_time is not None
                else elapsed
            ),
        }

        existing_names = {
            item["name"]
            for item in st.session_state.uploaded_files
        }

        if uploaded_file.name not in existing_names:
            st.session_state.uploaded_files.append(file_info)
            save_uploaded_files_list()

        return (
            True,
            (
                f"✅ Successfully processed {uploaded_file.name} "
                f"({len(result.chunks)} chunks, {elapsed:.2f}s)"
            ),
        )

    except Exception as exc:
        return (
            False,
            f"❌ Error processing {uploaded_file.name}: {exc}",
        )

    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def process_multiple_files(
    uploaded_files,
    rag_system,
    progress_bar,
    status_text,
):
    """Process multiple uploaded files."""
    successful = 0
    failed = 0

    for index, uploaded_file in enumerate(uploaded_files):
        status_text.info(f"📄 Processing: {uploaded_file.name}")

        try:
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()

            stored = store_file_in_db(
                filename=uploaded_file.name,
                file_content=file_content,
                file_type=uploaded_file.type,
                file_size=uploaded_file.size,
                file_hash=file_hash,
                upload_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            if not stored:
                failed += 1
                status_text.error(
                    f"❌ Could not store {uploaded_file.name}"
                )
            else:
                uploaded_file.seek(0)
                success, message = process_uploaded_file(
                    uploaded_file,
                    rag_system,
                )

                if success:
                    successful += 1
                    status_text.success(message)
                else:
                    failed += 1
                    status_text.error(message)

        except Exception as exc:
            failed += 1
            status_text.error(
                f"❌ Error processing {uploaded_file.name}: {exc}"
            )

        progress_bar.progress(
            (index + 1) / len(uploaded_files)
        )

    return successful, failed


# ============================================================
# FILE MANAGEMENT
# ============================================================
def clear_recent_uploads() -> None:
    """Hide recent uploads and clear session metadata."""
    st.session_state.uploaded_files = []
    st.session_state.hide_recent_uploads = True
    save_uploaded_files_list()
    st.success("Recent uploads list cleared.")


def show_recent_uploads() -> None:
    """Restore recent uploads view."""
    st.session_state.hide_recent_uploads = False
    st.success("Recent uploads restored.")


def clear_all_data() -> None:
    """Clear chat and study-session state."""
    st.session_state.messages = []
    st.session_state.uploaded_files = []
    st.session_state.study_summary = ""
    st.session_state.study_quiz = ""
    save_uploaded_files_list()
    st.success("Chat and study-session data cleared.")


# ============================================================
# STUDY TOOLS
# ============================================================
def render_study_tools() -> None:
    """Render summary and quiz generation tools."""
    st.markdown(
        '<div class="study-tools">',
        unsafe_allow_html=True,
    )

    st.subheader("📚 Study Tools")
    st.caption(
        "Generate revision material from the documents indexed in ChromaDB."
    )

    topic = st.text_input(
        "Topic",
        value=(
            ""
            if st.session_state.summary_topic == "all"
            else st.session_state.summary_topic
        ),
        placeholder=(
            "Leave empty for the entire indexed content, "
            "or enter a specific topic..."
        ),
        key="summary_topic_input",
    )

    col1, col2 = st.columns(2)

    with col1:
        generate_summary_clicked = st.button(
            "📖 Generate Summary",
            type="primary",
            use_container_width=True,
            key="generate_summary_btn",
        )

    with col2:
        generate_quiz_clicked = st.button(
            "📝 Generate Quiz",
            type="primary",
            use_container_width=True,
            key="generate_quiz_btn",
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    if generate_summary_clicked:
        if not st.session_state.uploaded_files:
            st.warning(
                "Please upload and process at least one document first."
            )
        else:
            rag_system = get_rag_system()
            if rag_system is None:
                st.error("SmartRAG system is unavailable.")
            else:
                summary_topic = topic.strip() or "all"
                st.session_state.summary_topic = summary_topic
                st.session_state.study_quiz = ""

                with st.spinner(
                    "📖 Generating grounded study summary..."
                ):
                    start_time = time.time()
                    summary = rag_system.generate_summary(
                        topic=summary_topic,
                        max_chunks=8,
                    )
                    elapsed = time.time() - start_time

                if summary and not summary.startswith("Error"):
                    st.session_state.study_summary = summary
                    st.success(
                        f"✅ Summary generated in {elapsed:.2f}s"
                    )
                else:
                    st.error(
                        summary or "Summary generation failed."
                    )

    # --------------------------------------------------------
    # QUIZ
    # --------------------------------------------------------
    if generate_quiz_clicked:
        if not st.session_state.uploaded_files:
            st.warning(
                "Please upload and process at least one document first."
            )
        else:
            rag_system = get_rag_system()
            if rag_system is None:
                st.error("SmartRAG system is unavailable.")
            else:
                quiz_topic = topic.strip() or "all"
                st.session_state.summary_topic = quiz_topic
                st.session_state.study_summary = ""

                with st.spinner(
                    "📝 Generating grounded quiz..."
                ):
                    start_time = time.time()
                    quiz = rag_system.generate_quiz(
                        topic=quiz_topic,
                        num_questions=5,
                        max_chunks=8,
                    )
                    elapsed = time.time() - start_time

                if quiz and not quiz.startswith("Error"):
                    st.session_state.study_quiz = quiz
                    st.success(
                        f"✅ Quiz generated in {elapsed:.2f}s"
                    )
                else:
                    st.error(
                        quiz or "Quiz generation failed."
                    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SUMMARY RESULT
    # --------------------------------------------------------
    if st.session_state.study_summary:
        st.markdown(
            '<div class="study-summary">',
            unsafe_allow_html=True,
        )

        st.markdown("## 📖 Study Summary")
        st.markdown(st.session_state.study_summary)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📥 Download Summary",
                data=st.session_state.study_summary,
                file_name="study_summary.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_summary",
            )

        with col2:
            if st.button(
                "🗑️ Clear Summary",
                use_container_width=True,
                key="clear_summary",
            ):
                st.session_state.study_summary = ""
                st.rerun()

    # --------------------------------------------------------
    # QUIZ RESULT
    # --------------------------------------------------------
    if st.session_state.study_quiz:
        st.markdown(
            '<div class="quiz-card">',
            unsafe_allow_html=True,
        )

        st.markdown("## 📝 Generated Quiz")
        st.markdown(st.session_state.study_quiz)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📥 Download Quiz",
                data=st.session_state.study_quiz,
                file_name="study_quiz.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_quiz",
            )

        with col2:
            if st.button(
                "🗑️ Clear Quiz",
                use_container_width=True,
                key="clear_quiz",
            ):
                st.session_state.study_quiz = ""
                st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================
def render_custom_chat_input():
    """Render chat input and audio transcription UI."""
    col1, col2, col3 = st.columns([1, 10, 1])

    with col1:
        if WHISPER_AVAILABLE:
            if st.button(
                "🎙️",
                help="Upload audio for speech-to-text",
                key="stt_btn",
            ):
                st.session_state.show_audio_upload = True
        else:
            st.button(
                "🎙️",
                disabled=True,
                help="Install openai-whisper.",
                key="stt_disabled",
            )

    user_text = None

    with col2:
        message_count = len(st.session_state.messages)

        with st.form(
            key=f"chat_form_{message_count}",
            clear_on_submit=True,
        ):
            text_message = st.text_input(
                "message",
                placeholder="Ask a question about your documents...",
                label_visibility="collapsed",
                key=f"chat_input_{message_count}",
            )

            send_clicked = st.form_submit_button(
                "Send",
                use_container_width=True,
            )

            if send_clicked and text_message.strip():
                user_text = text_message.strip()

    with col3:
        st.markdown(
            '<div style="text-align:center; padding:8px;">📝</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.show_audio_upload:
        with st.expander(
            "🎙️ Speech-to-Text",
            expanded=True,
        ):
            audio_file = st.file_uploader(
                "Choose audio",
                type=[
                    "wav",
                    "mp3",
                    "m4a",
                    "ogg",
                    "flac",
                    "webm",
                ],
                key=f"speech_upload_{message_count}",
            )

            col_a, col_b = st.columns(2)

            with col_a:
                if audio_file and st.button(
                    "🎯 Transcribe",
                    key="transcribe_btn",
                ):
                    transcript = transcribe_audio_file(
                        audio_file
                    )

                    if transcript:
                        st.session_state.show_audio_upload = False
                        st.session_state.transcribed_message = transcript
                        st.rerun()

            with col_b:
                if st.button(
                    "❌ Cancel",
                    key="cancel_stt",
                ):
                    st.session_state.show_audio_upload = False
                    st.rerun()

    if (
        not user_text
        and st.session_state.transcribed_message
    ):
        user_text = st.session_state.transcribed_message
        st.session_state.transcribed_message = ""

    return user_text


# ============================================================
# SOURCE UTILITIES
# ============================================================
def extract_source_names(response) -> list:
    """Extract unique source filenames from a query response."""
    names = set()

    for source in response.sources:
        source_file = (
            source.source_file
            or source.metadata.get("filename", "Unknown")
        )

        if not source_file:
            continue

        filename = (
            str(source_file)
            .replace("\\", "/")
            .split("/")[-1]
        )

        if filename:
            names.add(filename)

    return sorted(names)


# ============================================================
# SIDEBAR
# ============================================================
def sidebar_content() -> None:
    """Render the application sidebar."""
    with st.sidebar:
        st.title("🤖 SmartRAG")
        st.caption("Offline Multimodal Study Assistant")
        st.markdown("---")

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------
        st.subheader("📊 System Status")

        if st.session_state.system_initialized:
            st.success("🟢 System Ready")

            stats = {}
            try:
                stats = (
                    st.session_state.rag_system
                    .get_system_stats()
                )
            except Exception:
                pass

            st.metric(
                "Documents",
                len(st.session_state.uploaded_files),
            )

            st.metric(
                "LLM",
                (
                    "✅ Ready"
                    if stats.get("llm_available", False)
                    else "❌ Offline"
                ),
            )

            if stats.get("summary_available", False):
                st.info("📖 Study summary available")

            if stats.get("quiz_available", False):
                st.info("📝 Quiz generation available")

            if WHISPER_AVAILABLE:
                st.info("🎙️ Whisper available (CPU)")

        else:
            st.error("🔴 System Not Ready")

            if st.button(
                "🔄 Retry Initialization",
                key="retry_init",
            ):
                st.session_state.rag_system = None
                st.session_state.system_initialized = False
                st.rerun()

        st.markdown("---")

        # ----------------------------------------------------
        # RECENT UPLOADS
        # ----------------------------------------------------
        st.subheader("🕒 Recent Uploads")

        if st.session_state.hide_recent_uploads:
            st.caption("Recent uploads hidden.")

            if st.button(
                "👁️ Show",
                key="show_uploads",
            ):
                show_recent_uploads()
                st.rerun()
        else:
            recent_files = get_all_stored_files()[:5]

            if recent_files:
                for (
                    filename,
                    file_type,
                    file_size,
                    upload_time,
                ) in recent_files:
                    display_name = (
                        filename
                        if len(filename) <= 24
                        else f"{filename[:21]}..."
                    )

                    st.markdown(
                        f"📄 **{display_name}**"
                    )

                    st.caption(
                        f"{file_size // 1024} KB | {upload_time}"
                    )
            else:
                st.caption("No uploads yet.")

        st.markdown("---")

        # ----------------------------------------------------
        # UPLOAD
        # ----------------------------------------------------
        with st.expander(
            "📁 Upload Documents",
            expanded=True,
        ):
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=[
                    "pdf",
                    "docx",
                    "doc",
                    "txt",
                    "md",
                    "rtf",
                    "jpg",
                    "jpeg",
                    "png",
                    "bmp",
                    "tiff",
                    "webp",
                    "mp3",
                    "wav",
                    "m4a",
                    "ogg",
                    "flac",
                    "aac",
                ],
                help=(
                    "PDF, DOCX, TXT, MD, images and audio."
                ),
                key="main_file_uploader",
            )

            if (
                uploaded_files
                and st.session_state.system_initialized
            ):
                if st.button(
                    "📤 Process Files",
                    type="primary",
                    use_container_width=True,
                    key="process_files_btn",
                ):
                    rag_system = get_rag_system()

                    if rag_system:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        successful, failed = process_multiple_files(
                            uploaded_files,
                            rag_system,
                            progress_bar,
                            status_text,
                        )

                        if failed == 0:
                            status_text.success(
                                f"✅ Successfully processed {successful} file(s)."
                            )
                        else:
                            status_text.warning(
                                f"⚠️ {successful} succeeded, {failed} failed."
                            )

        st.markdown("---")

        # ----------------------------------------------------
        # STORED FILES
        # ----------------------------------------------------
        db_files = get_all_stored_files()

        if db_files:
            with st.expander(
                f"📋 Stored Files ({len(db_files)})",
                expanded=False,
            ):
                for index, (
                    filename,
                    file_type,
                    file_size,
                    upload_time,
                ) in enumerate(db_files):
                    col1, col2, col3 = st.columns([4, 1, 1])

                    with col1:
                        st.write(f"📄 **{filename}**")
                        st.caption(
                            f"{file_size:,} bytes | {upload_time}"
                        )

                    with col2:
                        if st.button(
                            "👁️",
                            key=f"view_file_{index}",
                        ):
                            st.session_state.viewing_file = filename
                            st.session_state.show_file_viewer = True
                            st.rerun()

                    with col3:
                        if st.button(
                            "🗑️",
                            key=f"delete_file_{index}",
                        ):
                            if delete_file_from_db(filename):
                                st.success(
                                    f"Deleted {filename}."
                                )
                            st.rerun()

        st.markdown("---")

        # ----------------------------------------------------
        # DATA MANAGEMENT
        # ----------------------------------------------------
        st.subheader("🧹 Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Clear Chat",
                use_container_width=True,
                key="clear_chat",
            ):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button(
                "Hide Uploads",
                use_container_width=True,
                key="hide_uploads",
            ):
                clear_recent_uploads()
                st.rerun()

        col3, col4 = st.columns(2)

        with col3:
            if st.button(
                "Clear DB",
                use_container_width=True,
                key="clear_db",
            ):
                clear_database_files()
                st.rerun()

        with col4:
            if st.button(
                "Clear All",
                use_container_width=True,
                key="clear_all",
            ):
                clear_all_data()
                st.rerun()


# ============================================================
# EMERGENCY UPLOAD PANEL
# ============================================================
def render_emergency_sidebar() -> None:
    """Render backup upload panel."""
    if not st.session_state.show_emergency_sidebar:
        return

    with st.container():
        st.markdown("### 🔧 Backup Upload Panel")

        col_a, col_b, col_c = st.columns([1, 2, 1])

        with col_b:
            emergency_upload = st.file_uploader(
                "Choose files",
                type=[
                    "pdf",
                    "docx",
                    "txt",
                    "png",
                    "jpg",
                    "jpeg",
                    "mp3",
                    "wav",
                ],
                accept_multiple_files=True,
                key="emergency_upload",
            )

            if emergency_upload:
                st.success(
                    f"{len(emergency_upload)} file(s) ready."
                )

                if st.button(
                    "📤 Process Uploaded Files",
                    type="primary",
                    key="emergency_process_files",
                ):
                    rag_system = get_rag_system()

                    if rag_system is None:
                        st.error("SmartRAG is unavailable.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        successful, failed = process_multiple_files(
                            emergency_upload,
                            rag_system,
                            progress_bar,
                            status_text,
                        )

                        if failed == 0:
                            status_text.success(
                                f"✅ {successful} file(s) processed successfully."
                            )
                        else:
                            status_text.warning(
                                f"⚠️ {successful} succeeded, {failed} failed."
                            )

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Clear Chat",
                    key="emergency_clear_chat",
                ):
                    st.session_state.messages = []
                    st.rerun()

            with col2:
                if st.button(
                    "Close",
                    key="emergency_close",
                ):
                    st.session_state.show_emergency_sidebar = False
                    st.rerun()


# ============================================================
# CHAT INTERFACE
# ============================================================
def main_chat_interface() -> None:
    """Render the main application."""

    # --------------------------------------------------------
    # FILE VIEWER
    # --------------------------------------------------------
    if (
        st.session_state.show_file_viewer
        and st.session_state.viewing_file
    ):
        render_file_viewer()
        return

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------
    col1, col2, col3 = st.columns([1, 7, 1])

    with col1:
        if st.button(
            "🔧",
            help="Open backup upload tools",
            key="sidebar_tools_toggle",
        ):
            st.session_state.show_emergency_sidebar = (
                not st.session_state.show_emergency_sidebar
            )

    with col2:
        st.title("Chat with your Documents")
        st.markdown(
            "*Smart conversations with text, images and audio*"
        )

    with col3:
        if st.session_state.system_initialized:
            st.markdown("🟢 **Ready**")
        else:
            st.markdown("🔴 **Loading**")

    # --------------------------------------------------------
    # STUDY TOOLS
    # --------------------------------------------------------
    render_study_tools()

    # --------------------------------------------------------
    # BACKUP PANEL
    # --------------------------------------------------------
    render_emergency_sidebar()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="assistant-message">
                    <strong>SmartRAG:</strong><br>
                    Welcome! Upload your learning material and ask questions about it.
                    <br><br>
                    <small>Local RAG • Qwen • ChromaDB</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for index, message in enumerate(
                st.session_state.messages
            ):
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="user-message">
                            <strong>You:</strong><br>
                            {message["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    content = message.get("content", "")
                    metadata = message.get("metadata", "")

                    st.markdown(
                        f"""
                        <div class="assistant-message">
                            <strong>SmartRAG:</strong><br>
                            {content}
                            <br><br>
                            <small>{metadata}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Source buttons
                    source_pattern = (
                        r"(?:Source:|from)\s+"
                        r"([^\s,;.!?]+\.(?:pdf|docx?|txt|md|rtf|png|"
                        r"jpe?g|gif|bmp|tiff?|webp|wav|mp3|m4a|ogg|flac))"
                    )

                    sources = re.findall(
                        source_pattern,
                        content + " " + metadata,
                        re.IGNORECASE,
                    )

                    unique_sources = list(
                        dict.fromkeys(sources)
                    )

                    if unique_sources:
                        st.markdown("**📎 View Sources:**")

                        columns = st.columns(
                            min(len(unique_sources), 3)
                        )

                        for source_index, filename in enumerate(
                            unique_sources
                        ):
                            with columns[
                                source_index % len(columns)
                            ]:
                                if st.button(
                                    f"📄 {filename}",
                                    key=(
                                        f"source_{index}_{source_index}"
                                    ),
                                ):
                                    st.session_state.viewing_file = filename
                                    st.session_state.show_file_viewer = True
                                    st.rerun()

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------
    thinking_placeholder = st.empty()

    if not st.session_state.system_initialized:
        st.warning(
            "🚫 SmartRAG system is not initialized."
        )
        return

    user_input = render_custom_chat_input()

    if not user_input:
        return

    # --------------------------------------------------------
    # PREVENT DUPLICATES
    # --------------------------------------------------------
    input_hash = hashlib.md5(
        f"{user_input}_{len(st.session_state.messages)}".encode("utf-8")
    ).hexdigest()

    if input_hash == st.session_state.input_hash:
        return

    st.session_state.input_hash = input_hash

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------
    with thinking_placeholder.container():
        with st.spinner("🤔 Thinking..."):
            try:
                # Timestamp query
                timestamp_patterns = [
                    r"what.*uploaded.*at\s*(\d{1,2}[:.]\d{2})",
                    r"documents.*uploaded.*(\d{1,2}[:.]\d{2})",
                    r"files.*uploaded.*(\d{1,2}[:.]\d{2})",
                    r"uploaded.*at\s*(\d{1,2}[:.]\d{2})",
                ]

                timestamp_response = None

                for pattern in timestamp_patterns:
                    match = re.search(
                        pattern,
                        user_input.lower(),
                    )

                    if match:
                        timestamp_response = search_files_by_timestamp(
                            match.group(1)
                        )
                        break

                if timestamp_response:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": timestamp_response,
                            "metadata": "Upload timestamp search",
                        }
                    )

                else:
                    rag_system = get_rag_system()

                    if rag_system is None:
                        raise RuntimeError(
                            "SmartRAG system is unavailable."
                        )

                    query_request = QueryRequest(
                        query=user_input,
                        top_k=5,
                        include_metadata=True,
                    )

                    start_time = time.time()

                    response = rag_system.query(
                        query_request
                    )

                    elapsed = time.time() - start_time

                    source_names = extract_source_names(
                        response
                    )

                    metadata = (
                        "Sources: "
                        + ", ".join(source_names)
                        if source_names
                        else "Sources: No documents referenced"
                    )

                    metadata += (
                        f" | Response time: {elapsed:.2f}s"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "metadata": metadata,
                        }
                    )

            except Exception as exc:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Sorry, I encountered an error: "
                            f"{exc}"
                        ),
                        "metadata": "",
                    }
                )

    thinking_placeholder.empty()
    st.rerun()


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    """Application entry point."""
    init_session_state()
    load_uploaded_files_list()

    if (
        not st.session_state.system_initialized
        and st.session_state.rag_system is None
    ):
        get_rag_system()

    sidebar_content()
    main_chat_interface()


if __name__ == "__main__":
    main()