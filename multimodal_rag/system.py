
"""
Main multimodal RAG system implementation.

SmartRAG provides:
- PDF / DOCX / TXT / Markdown ingestion
- Image and audio processing
- Ollama-based local LLM inference
- ChromaDB semantic retrieval
- RAG question answering
- Document-specific grounded summarization
- Document-specific grounded quiz generation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from config_schema import SmartRAGConfig, load_config

    USE_NEW_CONFIG = True

except ImportError:
    SmartRAGConfig = Any
    USE_NEW_CONFIG = False

    logging.warning(
        "config_schema not found; using legacy configuration loading"
    )

from .base import (
    QueryRequest,
    QueryResponse,
    DocumentChunk,
    ProcessingResult,
    OllamaLLM,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

GLOBAL_KEYWORDS = {
    "all",
    "everything",
    "entire document",
    "whole document",
    "full document",
}

MAX_QUERY_TOP_K = 8
MAX_SUMMARY_CHUNKS = 12
MAX_QUIZ_CHUNKS = 12
MAX_GENERATION_TOKENS = 1024


# ============================================================
# SIMPLE RAG SYSTEM
# ============================================================

class SimpleRAGSystem:
    """
    Core RAG implementation.

    Supports:
    - File ingestion
    - ChromaDB retrieval
    - RAG question answering
    - Document-specific summaries
    - Document-specific quizzes
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], SmartRAGConfig],
    ):
        self._typed_config = None

        if (
            USE_NEW_CONFIG
            and not isinstance(config, dict)
            and hasattr(config, "to_dict")
        ):
            self.config = config.to_dict()
            self._typed_config = config
        else:
            self.config = config

        logger.info("Initializing Simple RAG System...")

        self.llm = None
        self.document_processor = None
        self.image_processor = None
        self.audio_processor = None
        self.vector_store = None

        try:
            # ------------------------------------------------
            # LLM
            # ------------------------------------------------

            self.llm = OllamaLLM(self.config)

            logger.info("Ollama LLM initialized")

            # ------------------------------------------------
            # PROCESSORS
            # ------------------------------------------------

            from .processors import (
                DocumentProcessorManager,
                ImageProcessorManager,
                AudioProcessorManager,
            )

            self.document_processor = (
                DocumentProcessorManager(self.config)
            )

            self.image_processor = (
                ImageProcessorManager(self.config)
            )

            self.audio_processor = (
                AudioProcessorManager(self.config)
            )

            # ------------------------------------------------
            # VECTOR STORE
            # ------------------------------------------------

            from .vector_stores.chroma_store import (
                ChromaVectorStore
            )

            self.vector_store = ChromaVectorStore(
                self.config
            )

            logger.info(
                "Simple RAG System initialized successfully"
            )

        except Exception as exc:

            logger.exception(
                "Failed to initialize Simple RAG System: %s",
                exc,
            )

            self.llm = None
            self.document_processor = None
            self.image_processor = None
            self.audio_processor = None
            self.vector_store = None

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(self) -> bool:
        """Return True when the local LLM is available."""

        if self.llm is None:
            return False

        if not hasattr(self.llm, "is_available"):
            return False

        try:
            return bool(self.llm.is_available())

        except Exception as exc:

            logger.error(
                "Error checking LLM availability: %s",
                exc,
            )

            return False

    # ========================================================
    # INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path: Union[str, Path],
    ) -> ProcessingResult:
        """Process and index a single file."""

        if not self.is_available():

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message="System not available",
            )

        path = Path(file_path)

        try:

            logger.info(
                "Ingesting file: %s",
                path,
            )

            result = None

            # ------------------------------------------------
            # DOCUMENT
            # ------------------------------------------------

            if (
                self.document_processor
                and self.document_processor.can_process(path)
            ):

                result = (
                    self.document_processor.process_file(
                        path
                    )
                )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            elif (
                self.image_processor
                and self.image_processor.can_process(path)
            ):

                result = (
                    self.image_processor.extract_content(
                        path
                    )
                )

            # ------------------------------------------------
            # AUDIO
            # ------------------------------------------------

            elif (
                self.audio_processor
                and self.audio_processor.can_process(path)
            ):

                result = (
                    self.audio_processor.extract_content(
                        path
                    )
                )

            # ------------------------------------------------
            # UNSUPPORTED
            # ------------------------------------------------

            else:

                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=(
                        f"Unsupported file type: "
                        f"{path.suffix}"
                    ),
                )

            # ------------------------------------------------
            # STORE IN CHROMADB
            # ------------------------------------------------

            if (
                result
                and result.success
                and result.chunks
                and self.vector_store
            ):

                stored = (
                    self.vector_store.add_documents(
                        result.chunks
                    )
                )

                if not stored:

                    logger.warning(
                        "Vector store did not confirm "
                        "document insertion."
                    )

                logger.info(
                    "Stored %d chunks for %s",
                    len(result.chunks),
                    path.name,
                )

            return result

        except Exception as exc:

            logger.exception(
                "Error ingesting file %s: %s",
                path,
                exc,
            )

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=str(exc),
            )

    # ========================================================
    # NORMAL QUERY
    # ========================================================

    def query(
        self,
        query: Union[str, QueryRequest],
    ) -> QueryResponse:
        """Answer a normal RAG question."""

        if not self.is_available():

            query_text = (
                query
                if isinstance(query, str)
                else query.query
            )

            return QueryResponse(
                answer="System not available",
                sources=[],
                query=query_text,
                confidence_score=0.0,
            )

        query_text = ""

        try:

            # ------------------------------------------------
            # NORMALIZE REQUEST
            # ------------------------------------------------

            if isinstance(query, str):

                query_text = query

                query_obj = QueryRequest(
                    query=query
                )

            else:

                query_text = query.query

                query_obj = query

            logger.info(
                "Processing query: %s",
                query_text,
            )

            is_conversational = (
                self._is_conversational_query(
                    query_text
                )
            )

            context = ""

            sources: List[
                DocumentChunk
            ] = []

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            if (
                not is_conversational
                and self.vector_store
            ):

                configured_top_k = (
                    self.config
                    .get(
                        "retrieval",
                        {},
                    )
                    .get(
                        "top_k",
                        5,
                    )
                )

                top_k = (
                    query_obj.top_k
                    if query_obj.top_k
                    else configured_top_k
                )

                top_k = max(
                    1,
                    min(
                        int(top_k),
                        MAX_QUERY_TOP_K,
                    ),
                )

                retrieval_result = (
                    self.vector_store.similarity_search(
                        query_text,
                        k=top_k,
                    )
                )

                relevant_chunks = (
                    self._filter_relevant_context(
                        retrieval_result.chunks,
                        max_chunks=top_k,
                    )
                )

                if relevant_chunks:

                    context = (
                        self._build_context(
                            relevant_chunks
                        )
                    )

                    sources = (
                        relevant_chunks
                    )

                logger.info(
                    "Retrieved %d chunks",
                    len(relevant_chunks),
                )

            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            response_text = (
                self._generate_contextual_response(
                    query_text,
                    context,
                    is_conversational,
                    **query_obj.generation_params,
                )
            )

            return QueryResponse(
                answer=response_text,
                sources=sources,
                query=query_text,
                confidence_score=(
                    0.8 if sources else 0.9
                ),
            )

        except Exception as exc:

            logger.exception(
                "Error processing query: %s",
                exc,
            )

            return QueryResponse(
                answer=(
                    f"Error processing query: "
                    f"{exc}"
                ),
                sources=[],
                query=query_text,
                confidence_score=0.0,
            )

    # ========================================================
    # FILENAME HELPERS
    # ========================================================

    @staticmethod
    def _normalize_filename(
        filename: Optional[str],
    ) -> Optional[str]:
        """Convert a path or filename into a normalized name."""

        if not filename:
            return None

        return (
            str(filename)
            .replace("\\", "/")
            .split("/")[-1]
            .strip()
            .lower()
        )

    def _chunk_filename(
        self,
        chunk: DocumentChunk,
    ) -> Optional[str]:
        """Get normalized filename from a chunk."""

        metadata = (
            chunk.metadata
            or {}
        )

        source_file = (
            chunk.source_file
            or metadata.get("source_file")
            or metadata.get("filename")
        )

        return self._normalize_filename(
            source_file
        )

    # ========================================================
    # EXACT DOCUMENT RETRIEVAL
    # ========================================================

    def _get_file_chunks(
        self,
        filename: str,
    ) -> List[DocumentChunk]:
        """
        Get chunks belonging ONLY to the requested file.

        This bypasses semantic retrieval for global
        document summaries and reads ChromaDB directly.
        """

        if not self.vector_store:
            return []

        target = (
            self._normalize_filename(
                filename
            )
        )

        if not target:
            return []

        # ----------------------------------------------------
        # Direct Chroma collection access
        # ----------------------------------------------------

        try:

            collection = getattr(
                self.vector_store,
                "collection",
                None,
            )

            if collection is None:
                raise RuntimeError(
                    "Chroma collection unavailable"
                )

            data = collection.get(
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            documents = (
                data.get(
                    "documents",
                    []
                )
            )

            metadatas = (
                data.get(
                    "metadatas",
                    []
                )
            )

            ids = (
                data.get(
                    "ids",
                    []
                )
            )

            chunks: List[
                DocumentChunk
            ] = []

            for index, content in enumerate(
                documents
            ):

                if not content:
                    continue

                metadata = (
                    metadatas[index]
                    if index < len(metadatas)
                    and metadatas[index]
                    else {}
                )

                source_file = (
                    metadata.get(
                        "source_file"
                    )
                    or metadata.get(
                        "filename"
                    )
                )

                current_name = (
                    self._normalize_filename(
                        source_file
                    )
                )

                if current_name != target:
                    continue

                chunk_id = (
                    ids[index]
                    if index < len(ids)
                    else ""
                )

                chunks.append(
                    DocumentChunk(
                        content=content,
                        metadata=metadata,
                        document_type=metadata.get(
                            "document_type",
                            "unknown",
                        ),
                        chunk_id=chunk_id or "",
                        source_file=source_file,
                    )
                )

            # ------------------------------------------------
            # Sort by original chunk index
            # ------------------------------------------------

            def chunk_sort_key(
                chunk: DocumentChunk,
            ):

                value = (
                    chunk.metadata
                    .get(
                        "chunk_index",
                        0,
                    )
                )

                try:
                    return int(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    return 0

            chunks.sort(
                key=chunk_sort_key
            )

            chunks = (
                self._clean_chunks(
                    chunks
                )
            )

            logger.info(
                "Found %d chunks for file %s",
                len(chunks),
                filename,
            )

            return chunks

        except Exception as exc:

            logger.warning(
                "Direct file retrieval failed for %s: %s",
                filename,
                exc,
            )

        # ----------------------------------------------------
        # Fallback metadata-filtered search
        # ----------------------------------------------------

        try:

            retrieval_result = (
                self.vector_store.similarity_search(
                    query=filename,
                    k=MAX_SUMMARY_CHUNKS,
                    filter_dict={
                        "filename": filename
                    },
                )
            )

            chunks = (
                self._clean_chunks(
                    retrieval_result.chunks
                )
            )

            # Final safety filter.
            return [
                chunk
                for chunk in chunks
                if (
                    self._chunk_filename(chunk)
                    == target
                )
            ]

        except Exception as exc:

            logger.error(
                "Filename-filtered retrieval failed: %s",
                exc,
            )

            return []

    # ========================================================
    # CLEAN CHUNKS
    # ========================================================

    def _clean_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        """Remove empty chunks and duplicate chunks."""

        cleaned = []

        seen = set()

        for chunk in chunks:

            if not chunk:
                continue

            if not chunk.content:
                continue

            content = (
                chunk.content.strip()
            )

            if len(content) <= 30:
                continue

            identifier = (
                chunk.chunk_id
                or (
                    f"{self._chunk_filename(chunk)}:"
                    f"{content[:100]}"
                )
            )

            if identifier in seen:
                continue

            seen.add(identifier)

            cleaned.append(
                chunk
            )

        return cleaned

    # ========================================================
    # DOCUMENT-SPECIFIC SUMMARY
    # ========================================================

    def generate_summary(
        self,
        topic: str = "all",
        max_chunks: int = 8,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a grounded summary.

        When filename is provided:
            ONLY that document is summarized.

        When filename is omitted:
            Normal global RAG summarization is used.
        """

        if not self.is_available():
            return "System not available."

        if not self.vector_store:
            return "Vector store is not available."

        try:

            max_chunks = max(
                1,
                min(
                    int(max_chunks),
                    MAX_SUMMARY_CHUNKS,
                ),
            )

            topic_text = (
                topic.strip()
                if topic
                else "all"
            )

            is_global_topic = (
                topic_text.lower()
                in GLOBAL_KEYWORDS
            )

            # =================================================
            # FILE-SPECIFIC MODE
            # =================================================

            if filename:

                logger.info(
                    "Generating document-specific summary: %s",
                    filename,
                )

                file_chunks = (
                    self._get_file_chunks(
                        filename
                    )
                )

                if not file_chunks:

                    return (
                        f"No indexed content was found "
                        f"for '{filename}'."
                    )

                # ---------------------------------------------
                # Entire selected document
                # ---------------------------------------------

                if is_global_topic:

                    relevant_chunks = (
                        file_chunks[
                            :max_chunks
                        ]
                    )

                # ---------------------------------------------
                # Topic inside selected document
                # ---------------------------------------------

                else:

                    try:

                        filtered_result = (
                            self.vector_store
                            .similarity_search(
                                query=topic_text,
                                k=max(
                                    max_chunks * 2,
                                    8,
                                ),
                                filter_dict={
                                    "filename": filename
                                },
                            )
                        )

                        filtered_chunks = (
                            self._clean_chunks(
                                filtered_result.chunks
                            )
                        )

                        # Final filename safety check.
                        target = (
                            self._normalize_filename(
                                filename
                            )
                        )

                        relevant_chunks = [
                            chunk
                            for chunk in filtered_chunks
                            if (
                                self._chunk_filename(
                                    chunk
                                )
                                == target
                            )
                        ][
                            :max_chunks
                        ]

                    except Exception as exc:

                        logger.warning(
                            "Topic-filtered file search failed: %s",
                            exc,
                        )

                        relevant_chunks = (
                            file_chunks[
                                :max_chunks
                            ]
                        )

            # =================================================
            # GLOBAL MODE
            # =================================================

            else:

                if is_global_topic:

                    search_query = (
                        "main topic "
                        "key concepts "
                        "technical details "
                        "architecture "
                        "process "
                        "methods "
                        "implementation "
                        "advantages "
                        "challenges "
                        "limitations "
                        "important findings "
                        "conclusions"
                    )

                else:

                    search_query = (
                        topic_text
                    )

                candidate_count = min(
                    max(
                        max_chunks * 2,
                        12,
                    ),
                    20,
                )

                retrieval_result = (
                    self.vector_store
                    .similarity_search(
                        search_query,
                        k=candidate_count,
                    )
                )

                if is_global_topic:

                    relevant_chunks = (
                        self._select_summary_chunks(
                            retrieval_result.chunks,
                            max_chunks=max_chunks,
                        )
                    )

                else:

                    relevant_chunks = (
                        self._filter_relevant_context(
                            retrieval_result.chunks,
                            max_chunks=max_chunks,
                        )
                    )

            # ------------------------------------------------
            # NO CONTENT
            # ------------------------------------------------

            if not relevant_chunks:

                return (
                    "No relevant document content "
                    "was found for summarization."
                )

            # ------------------------------------------------
            # BUILD CONTEXT
            # ------------------------------------------------

            context = (
                self._build_context(
                    relevant_chunks
                )
            )

            source_label = (
                filename
                if filename
                else "indexed documents"
            )

            # ------------------------------------------------
            # GROUNDED PROMPT
            # ------------------------------------------------

            prompt = f"""
You are EduSense AI, an offline AI study assistant.

Create a factually grounded study summary.

SELECTED SOURCE:
{source_label}

STRICT RULES:

1. Use ONLY information contained in the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Do NOT speculate.
5. Do NOT guess missing information.
6. Do NOT mix information from other documents.
7. Preserve technical names and numerical values.
8. Avoid repeating the same fact.
9. Keep the summary useful for student revision.
10. Use concise bullet points.
11. Omit sections that are unsupported.

Possible sections:

## Topic / Overview

## Key Concepts

## Important Technical Details

## Architecture / Process

## Advantages

## Challenges / Limitations

## Important Takeaways

DOCUMENT CONTEXT:

{context}

Generate the grounded summary now.
"""

            generation = (
                self.config.get(
                    "generation",
                    {}
                )
            )

            max_tokens = min(
                int(
                    generation.get(
                        "max_tokens",
                        512,
                    )
                ),
                MAX_GENERATION_TOKENS,
            )

            summary = (
                self.llm.generate_response(
                    prompt=prompt,
                    context="",
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=max_tokens,
                )
            )

            return summary.strip()

        except Exception as exc:

            logger.exception(
                "Error generating summary: %s",
                exc,
            )

            return (
                f"Error generating summary: {exc}"
            )

    # ========================================================
    # DOCUMENT-SPECIFIC QUIZ
    # ========================================================

    def generate_quiz(
        self,
        topic: str = "all",
        num_questions: int = 5,
        max_chunks: int = 8,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a grounded quiz.

        When filename is provided:
            ONLY that document is used.
        """

        if not self.is_available():
            return "System not available."

        if not self.vector_store:
            return "Vector store is not available."

        try:

            num_questions = max(
                1,
                min(
                    int(num_questions),
                    10,
                ),
            )

            max_chunks = max(
                1,
                min(
                    int(max_chunks),
                    MAX_QUIZ_CHUNKS,
                ),
            )

            topic_text = (
                topic.strip()
                if topic
                else "all"
            )

            is_global_topic = (
                topic_text.lower()
                in GLOBAL_KEYWORDS
            )

            # =================================================
            # FILE-SPECIFIC MODE
            # =================================================

            if filename:

                logger.info(
                    "Generating document-specific quiz: %s",
                    filename,
                )

                file_chunks = (
                    self._get_file_chunks(
                        filename
                    )
                )

                if not file_chunks:

                    return (
                        f"No indexed content was found "
                        f"for '{filename}'."
                    )

                if is_global_topic:

                    relevant_chunks = (
                        file_chunks[
                            :max_chunks
                        ]
                    )

                else:

                    try:

                        filtered_result = (
                            self.vector_store
                            .similarity_search(
                                query=topic_text,
                                k=max(
                                    max_chunks * 2,
                                    8,
                                ),
                                filter_dict={
                                    "filename": filename
                                },
                            )
                        )

                        filtered_chunks = (
                            self._clean_chunks(
                                filtered_result.chunks
                            )
                        )

                        target = (
                            self._normalize_filename(
                                filename
                            )
                        )

                        relevant_chunks = [
                            chunk
                            for chunk in filtered_chunks
                            if (
                                self._chunk_filename(
                                    chunk
                                )
                                == target
                            )
                        ][
                            :max_chunks
                        ]

                    except Exception as exc:

                        logger.warning(
                            "Topic-filtered quiz search failed: %s",
                            exc,
                        )

                        relevant_chunks = (
                            file_chunks[
                                :max_chunks
                            ]
                        )

            # =================================================
            # GLOBAL MODE
            # =================================================

            else:

                if is_global_topic:

                    search_query = (
                        "important concepts "
                        "important facts "
                        "key terminology "
                        "technical details "
                        "architecture "
                        "process "
                        "methods "
                        "implementation"
                    )

                else:

                    search_query = (
                        topic_text
                    )

                candidate_count = min(
                    max(
                        max_chunks * 2,
                        12,
                    ),
                    20,
                )

                retrieval_result = (
                    self.vector_store
                    .similarity_search(
                        search_query,
                        k=candidate_count,
                    )
                )

                if is_global_topic:

                    relevant_chunks = (
                        self._select_summary_chunks(
                            retrieval_result.chunks,
                            max_chunks=max_chunks,
                        )
                    )

                else:

                    relevant_chunks = (
                        self._filter_relevant_context(
                            retrieval_result.chunks,
                            max_chunks=max_chunks,
                        )
                    )

            # ------------------------------------------------
            # NO CONTENT
            # ------------------------------------------------

            if not relevant_chunks:

                return (
                    "No relevant document content "
                    "was found for quiz generation."
                )

            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            context = (
                self._build_context(
                    relevant_chunks
                )
            )

            source_label = (
                filename
                if filename
                else "indexed documents"
            )

            # ------------------------------------------------
            # QUIZ PROMPT
            # ------------------------------------------------

            prompt = f"""
You are EduSense AI, an offline AI study assistant.

Generate a multiple-choice quiz ONLY from the
supplied document context.

SELECTED SOURCE:
{source_label}

NUMBER OF QUESTIONS:
{num_questions}

STRICT RULES:

1. Use ONLY information in the context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Do NOT mix information from another document.
5. Every question must be answerable from the context.
6. Each question must have exactly four options.
7. Options must be A, B, C and D.
8. Exactly one option must be correct.
9. The Answer letter must match the correct option.
10. The Explanation must match the correct option.
11. Do not create trick questions.
12. Avoid duplicate questions.
13. Preserve technical terminology.
14. Do not ask questions that are unsupported by the context.

FORMAT:

## 📝 Quiz

### Q1. Question

A. Option
B. Option
C. Option
D. Option

**Answer:** A

**Explanation:** Explanation supported directly by the document.

DOCUMENT CONTEXT:

{context}

Generate the quiz now.
"""

            generation = (
                self.config.get(
                    "generation",
                    {}
                )
            )

            max_tokens = min(
                int(
                    generation.get(
                        "max_tokens",
                        512,
                    )
                ),
                MAX_GENERATION_TOKENS,
            )

            quiz = (
                self.llm.generate_response(
                    prompt=prompt,
                    context="",
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=max_tokens,
                )
            )

            return quiz.strip()

        except Exception as exc:

            logger.exception(
                "Error generating quiz: %s",
                exc,
            )

            return (
                f"Error generating quiz: {exc}"
            )

    # ========================================================
    # SUMMARY CHUNK SELECTION
    # ========================================================

    def _select_summary_chunks(
        self,
        chunks: List[DocumentChunk],
        max_chunks: int = 8,
    ) -> List[DocumentChunk]:
        """Select diverse chunks from search results."""

        chunks = (
            self._clean_chunks(
                chunks
            )
        )

        if not chunks:
            return []

        max_chunks = max(
            1,
            min(
                int(max_chunks),
                len(chunks),
            ),
        )

        selected = []

        selected_ids = set()

        seen_pages = set()

        # ----------------------------------------------------
        # First pass: prefer different pages
        # ----------------------------------------------------

        for chunk in chunks:

            page_number = (
                chunk.page_number
                or chunk.metadata.get(
                    "page_number"
                )
            )

            if page_number is None:
                continue

            if page_number in seen_pages:
                continue

            selected.append(
                chunk
            )

            selected_ids.add(
                chunk.chunk_id
            )

            seen_pages.add(
                page_number
            )

            if len(selected) >= max_chunks:
                return selected

        # ----------------------------------------------------
        # Second pass: fill remaining slots
        # ----------------------------------------------------

        for chunk in chunks:

            if chunk.chunk_id in selected_ids:
                continue

            selected.append(
                chunk
            )

            selected_ids.add(
                chunk.chunk_id
            )

            if len(selected) >= max_chunks:
                break

        return selected

    # ========================================================
    # CONTEXT FILTER
    # ========================================================

    def _filter_relevant_context(
        self,
        chunks: List[DocumentChunk],
        max_chunks: int = 5,
    ) -> List[DocumentChunk]:
        """Remove unusable chunks."""

        cleaned = (
            self._clean_chunks(
                chunks
            )
        )

        return cleaned[
            :max_chunks
        ]

    # ========================================================
    # CONVERSATIONAL DETECTION
    # ========================================================

    def _is_conversational_query(
        self,
        query_text: str,
    ) -> bool:
        """Detect casual conversation."""

        conversational_patterns = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what is your name",
            "who are you",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "see you",
            "nice to meet you",
            "how do you do",
        ]

        query_lower = (
            query_text.lower().strip()
        )

        for pattern in conversational_patterns:

            if (
                query_lower == pattern
                or query_lower.startswith(
                    pattern
                )
            ):
                return True

        if (
            len(query_lower) <= 10
            and not any(
                marker in query_lower
                for marker in [
                    "?",
                    "what",
                    "how",
                    "when",
                    "where",
                    "why",
                ]
            )
        ):

            return True

        return False

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def _build_context(
        self,
        chunks: List[DocumentChunk],
    ) -> str:
        """Build context with source and page information."""

        parts = []

        for index, chunk in enumerate(
            chunks
        ):

            metadata = (
                chunk.metadata
                or {}
            )

            source_file = (
                chunk.source_file
                or metadata.get(
                    "source_file"
                )
                or metadata.get(
                    "filename",
                    "Unknown Document",
                )
            )

            source_name = (
                str(source_file)
                .replace("\\", "/")
                .split("/")[-1]
            )

            page_number = (
                chunk.page_number
                or metadata.get(
                    "page_number"
                )
            )

            page_info = (
                f" (Page {page_number})"
                if page_number is not None
                else ""
            )

            parts.append(
                (
                    f"[Chunk {index + 1}] "
                    f"Source: {source_name}"
                    f"{page_info}\n"
                    f"{chunk.content}"
                )
            )

        return "\n\n".join(
            parts
        )

    # ========================================================
    # RESPONSE GENERATION
    # ========================================================

    def _generate_contextual_response(
        self,
        query_text: str,
        context: str,
        is_conversational: bool,
        **kwargs,
    ) -> str:
        """Generate a normal assistant response."""

        if is_conversational:

            prompt = f"""
You are a helpful offline AI assistant.

Respond naturally and briefly.

User:
{query_text}

Assistant:
"""

        elif context.strip():

            prompt = f"""
You are a helpful offline AI study assistant.

Answer the user's question using the supplied
document context.

Rules:
- Use the context as the primary source.
- Do not invent document-specific facts.
- If the answer is not found, say so.
- Do not pretend outside knowledge came from the document.

DOCUMENT CONTEXT:

{context}

QUESTION:

{query_text}

ANSWER:
"""

        else:

            prompt = f"""
You are a helpful offline AI assistant.

No relevant document context was retrieved.

Answer the question generally.

QUESTION:

{query_text}

ANSWER:
"""

        safe_kwargs = dict(
            kwargs
        )

        if "max_tokens" in safe_kwargs:

            safe_kwargs["max_tokens"] = min(
                int(
                    safe_kwargs[
                        "max_tokens"
                    ]
                ),
                768,
            )

        return (
            self.llm.generate_response(
                prompt=prompt,
                context="",
                **safe_kwargs,
            )
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_system_status(
        self,
    ) -> Dict[str, Any]:
        """Return system health information."""

        try:

            return {
                "system_type":
                    "simple_traditional",

                "llm_available":
                    self.is_available(),

                "processors_available": {
                    "documents":
                        self.document_processor is not None,

                    "images":
                        self.image_processor is not None,

                    "audio":
                        self.audio_processor is not None,
                },

                "vector_store_available":
                    self.vector_store is not None,

                "summary_available":
                    True,

                "quiz_available":
                    True,

                "model_name":
                    self.config
                    .get(
                        "models",
                        {}
                    )
                    .get(
                        "llm_model",
                        "unknown",
                    ),

                "embedding_model":
                    self.config
                    .get(
                        "models",
                        {}
                    )
                    .get(
                        "embedding_model",
                        "unknown",
                    ),
            }

        except Exception as exc:

            logger.exception(
                "Failed to get system status: %s",
                exc,
            )

            return {
                "system_type":
                    "simple_traditional",

                "error":
                    str(exc),

                "llm_available":
                    False,

                "summary_available":
                    False,

                "quiz_available":
                    False,
            }


# ============================================================
# PUBLIC MULTIMODAL RAG WRAPPER
# ============================================================

class MultimodalRAGSystem:
    """
    Public SmartRAG interface.

    Uses EnhancedMultimodalRAGSystem when available,
    otherwise SimpleRAGSystem.
    """

    def __init__(
        self,
        config_path: Optional[
            Union[str, Path]
        ] = None,
        config_dict: Optional[
            Dict[str, Any]
        ] = None,
        **overrides,
    ):

        self.config = (
            self._load_configuration(
                config_path,
                config_dict,
                overrides,
            )
        )

        self._system = None

        self.system_type = "none"

        # ====================================================
        # ENHANCED BACKEND
        # ====================================================

        try:

            from .enhanced_system import (
                EnhancedMultimodalRAGSystem
            )

            enhanced_system = (
                EnhancedMultimodalRAGSystem(
                    self.config
                )
            )

            if enhanced_system.is_available():

                self._system = (
                    enhanced_system
                )

                self.system_type = (
                    "enhanced_traditional"
                )

                logger.info(
                    "Enhanced RAG system initialized"
                )

        except Exception as exc:

            logger.warning(
                "Enhanced backend unavailable: %s",
                exc,
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        if self._system is None:

            try:

                fallback = (
                    SimpleRAGSystem(
                        self.config
                    )
                )

                if fallback.is_available():

                    self._system = (
                        fallback
                    )

                    self.system_type = (
                        "simple_traditional"
                    )

                    logger.info(
                        "Simple RAG fallback initialized"
                    )

            except Exception as exc:

                logger.exception(
                    "Fallback initialization failed: %s",
                    exc,
                )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def _load_configuration(
        self,
        config_path,
        config_dict,
        overrides,
    ) -> Dict[str, Any]:

        if USE_NEW_CONFIG:

            try:

                typed_config = (
                    load_config(
                        config_path=config_path,
                        **overrides,
                    )
                )

                return (
                    typed_config.to_dict()
                )

            except Exception as exc:

                logger.warning(
                    "Validated configuration failed: %s",
                    exc,
                )

        if config_dict:
            return config_dict

        if config_path:
            return (
                self._load_config(
                    config_path
                )
            )

        return (
            self._get_default_config()
        )

    def _load_config(
        self,
        config_path,
    ) -> Dict[str, Any]:

        import yaml

        try:

            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as file:

                config = (
                    yaml.safe_load(file)
                )

            return config or {}

        except Exception as exc:

            logger.error(
                "Failed to load config: %s",
                exc,
            )

            return (
                self._get_default_config()
            )

    def _get_default_config(
        self,
    ) -> Dict[str, Any]:

        return {
            "system": {
                "name": "SmartRAG System",
                "offline_mode": True,
                "debug": False,
                "log_level": "INFO",
            },

            "models": {
                "llm_type": "ollama",
                "llm_model": "qwen2.5:3b",
                "ollama_host": (
                    "http://localhost:11434"
                ),
                "embedding_model": (
                    "nomic-embed-text"
                ),
                "embedding_dimension": 768,
                "vision_model": (
                    "Salesforce/"
                    "blip-image-captioning-base"
                ),
                "whisper_model": "base",
                "whisper_device": "cpu",
            },

            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./vector_db",
                "collection_name": (
                    "multimodal_documents"
                ),
                "embedding_dimension": 768,
                "ollama_host": (
                    "http://localhost:11434"
                ),
            },

            "processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "ocr_enabled": True,
                "batch_size": 16,
                "store_original_images": True,
                "image_preprocessing": "resize",
                "audio_sample_rate": 16000,
                "max_audio_duration": 300,
            },

            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "rerank_enabled": False,
            },

            "generation": {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "max_new_tokens": 512,
            },
        }

    # ========================================================
    # INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path: Union[str, Path],
    ) -> ProcessingResult:

        if self._system is None:

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=(
                    "No active RAG system"
                ),
            )

        return (
            self._system.ingest_file(
                file_path
            )
        )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(
        self,
    ) -> bool:

        return (
            self._system is not None
            and self._system.is_available()
        )

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        query: Union[str, QueryRequest],
    ) -> QueryResponse:

        if self._system is None:

            query_text = (
                query
                if isinstance(query, str)
                else query.query
            )

            return QueryResponse(
                answer=(
                    "No active RAG system. "
                    "Please check Ollama."
                ),
                sources=[],
                query=query_text,
                confidence_score=0.0,
            )

        return (
            self._system.query(
                query
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def generate_summary(
        self,
        topic: str = "all",
        max_chunks: int = 8,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a summary.

        filename:
            When supplied, ONLY that document is used.
        """

        if self._system is None:
            return "No active RAG system."

        # ----------------------------------------------------
        # Backend supports filename
        # ----------------------------------------------------

        try:

            return (
                self._system.generate_summary(
                    topic=topic,
                    max_chunks=max_chunks,
                    filename=filename,
                )
            )

        except TypeError:

            logger.warning(
                "Backend does not support filename "
                "argument directly."
            )

        # ----------------------------------------------------
        # Fallback for Enhanced backend
        # ----------------------------------------------------

        if filename:

            return (
                self._document_specific_summary_fallback(
                    topic=topic,
                    max_chunks=max_chunks,
                    filename=filename,
                )
            )

        # ----------------------------------------------------
        # Legacy behavior
        # ----------------------------------------------------

        if hasattr(
            self._system,
            "generate_summary",
        ):

            return (
                self._system.generate_summary(
                    topic=topic,
                    max_chunks=max_chunks,
                )
            )

        return (
            "Summary generation is not "
            "available in the active backend."
        )

    def _get_backend_components(self):
        """
        Safely retrieve the real LLM and vector store.

        Enhanced backend:
            _system._simple_system

        Simple backend:
            _system itself
        """

        backend = self._system

        # Enhanced backend
        simple = getattr(
            backend,
            "_simple_system",
            None,
        )

        if simple is not None:

            return (
                getattr(simple, "llm", None),
                getattr(simple, "vector_store", None),
            )

        # Simple backend
        return (
            getattr(backend, "llm", None),
            getattr(backend, "vector_store", None),
        )

    def _document_specific_summary_fallback(
        self,
        topic: str,
        max_chunks: int,
        filename: str,
    ) -> str:
        """
        Generate a summary directly from one document.

        Used only when the active backend does not expose
        the filename parameter.
        """

        llm, vector_store = (
            self._get_backend_components()
        )

        if llm is None:

            return (
                "LLM is unavailable."
            )

        if vector_store is None:

            return (
                "Vector store is unavailable."
            )

        # Build helper using the existing SimpleRAG logic.
        helper = (
            SimpleRAGSystem.__new__(
                SimpleRAGSystem
            )
        )

        helper.config = self.config
        helper.llm = llm
        helper.vector_store = vector_store

        chunks = (
            helper._get_file_chunks(
                filename
            )
        )

        if not chunks:

            return (
                f"No indexed content was found "
                f"for '{filename}'."
            )

        topic_text = (
            topic.strip()
            if topic
            else "all"
        )

        if (
            topic_text.lower()
            in GLOBAL_KEYWORDS
        ):

            chunks = (
                chunks[
                    :max_chunks
                ]
            )

        else:

            try:

                result = (
                    vector_store.similarity_search(
                        query=topic_text,
                        k=max(
                            max_chunks * 2,
                            8,
                        ),
                        filter_dict={
                            "filename": filename
                        },
                    )
                )

                target = (
                    helper._normalize_filename(
                        filename
                    )
                )

                filtered = (
                    helper._clean_chunks(
                        result.chunks
                    )
                )

                chunks = [
                    chunk
                    for chunk in filtered
                    if (
                        helper._chunk_filename(
                            chunk
                        )
                        == target
                    )
                ][
                    :max_chunks
                ]

            except Exception as exc:

                logger.warning(
                    "Fallback topic retrieval failed: %s",
                    exc,
                )

                chunks = (
                    chunks[
                        :max_chunks
                    ]
                )

        if not chunks:

            return (
                f"No relevant content was found "
                f"in '{filename}'."
            )

        context = (
            helper._build_context(
                chunks
            )
        )

        prompt = f"""
You are EduSense AI, an offline AI study assistant.

Summarize ONLY the following document:

{filename}

Use ONLY the supplied context.

Do not:
- use outside knowledge
- invent facts
- speculate
- mix information from another document
- guess missing information

Prefer concise sections and bullet points.

DOCUMENT CONTEXT:

{context}

Generate the grounded study summary.
"""

        try:

            return (
                llm.generate_response(
                    prompt=prompt,
                    context="",
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=768,
                )
                .strip()
            )

        except Exception as exc:

            return (
                f"Error generating summary: {exc}"
            )

    # ========================================================
    # QUIZ
    # ========================================================

    def generate_quiz(
        self,
        topic: str = "all",
        num_questions: int = 5,
        max_chunks: int = 8,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a quiz.

        filename:
            When supplied, ONLY that document is used.
        """

        if self._system is None:
            return "No active RAG system."

        try:

            return (
                self._system.generate_quiz(
                    topic=topic,
                    num_questions=num_questions,
                    max_chunks=max_chunks,
                    filename=filename,
                )
            )

        except TypeError:

            logger.warning(
                "Backend does not support filename "
                "argument directly."
            )

        if filename:

            return (
                self._document_specific_quiz_fallback(
                    topic=topic,
                    num_questions=num_questions,
                    max_chunks=max_chunks,
                    filename=filename,
                )
            )

        if hasattr(
            self._system,
            "generate_quiz",
        ):

            return (
                self._system.generate_quiz(
                    topic=topic,
                    num_questions=num_questions,
                    max_chunks=max_chunks,
                )
            )

        return (
            "Quiz generation is not "
            "available in the active backend."
        )

    def _document_specific_quiz_fallback(
        self,
        topic: str,
        num_questions: int,
        max_chunks: int,
        filename: str,
    ) -> str:
        """Generate a quiz directly from one document."""

        llm, vector_store = (
            self._get_backend_components()
        )

        if llm is None:

            return (
                "LLM is unavailable."
            )

        if vector_store is None:

            return (
                "Vector store is unavailable."
            )

        helper = (
            SimpleRAGSystem.__new__(
                SimpleRAGSystem
            )
        )

        helper.config = self.config
        helper.llm = llm
        helper.vector_store = vector_store

        chunks = (
            helper._get_file_chunks(
                filename
            )
        )

        if not chunks:

            return (
                f"No indexed content was found "
                f"for '{filename}'."
            )

        topic_text = (
            topic.strip()
            if topic
            else "all"
        )

        if (
            topic_text.lower()
            in GLOBAL_KEYWORDS
        ):

            chunks = (
                chunks[
                    :max_chunks
                ]
            )

        else:

            try:

                result = (
                    vector_store.similarity_search(
                        query=topic_text,
                        k=max(
                            max_chunks * 2,
                            8,
                        ),
                        filter_dict={
                            "filename": filename
                        },
                    )
                )

                target = (
                    helper._normalize_filename(
                        filename
                    )
                )

                filtered = (
                    helper._clean_chunks(
                        result.chunks
                    )
                )

                chunks = [
                    chunk
                    for chunk in filtered
                    if (
                        helper._chunk_filename(
                            chunk
                        )
                        == target
                    )
                ][
                    :max_chunks
                ]

            except Exception as exc:

                logger.warning(
                    "Fallback topic quiz retrieval failed: %s",
                    exc,
                )

                chunks = (
                    chunks[
                        :max_chunks
                    ]
                )

        if not chunks:

            return (
                f"No relevant content was found "
                f"in '{filename}'."
            )

        context = (
            helper._build_context(
                chunks
            )
        )

        prompt = f"""
You are EduSense AI, an offline AI study assistant.

Generate a multiple-choice quiz using ONLY:

{filename}

Generate {num_questions} questions.

Rules:

1. Use only the supplied document context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not mix another document.
5. Each question must have four options.
6. Options must be A, B, C and D.
7. Exactly one option must be correct.
8. Give the answer.
9. Give a short explanation.
10. Avoid duplicate questions.

Format:

## 📝 Quiz

### Q1. Question

A. Option
B. Option
C. Option
D. Option

**Answer:** A

**Explanation:** Explanation.

DOCUMENT CONTEXT:

{context}

Generate the quiz now.
"""

        try:

            return (
                llm.generate_response(
                    prompt=prompt,
                    context="",
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=1024,
                )
                .strip()
            )

        except Exception as exc:

            return (
                f"Error generating quiz: {exc}"
            )

    # ========================================================
    # STATS
    # ========================================================

    def get_system_stats(
        self,
    ) -> Dict[str, Any]:

        if (
            self._system is not None
            and hasattr(
                self._system,
                "get_system_status",
            )
        ):

            stats = (
                self._system.get_system_status()
            )

            stats[
                "wrapper_system_type"
            ] = self.system_type

            # Add diagnostic information.
            llm, vector_store = (
                self._get_backend_components()
            )

            stats[
                "llm_exposed"
            ] = (
                llm is not None
            )

            stats[
                "vector_store_exposed"
            ] = (
                vector_store is not None
            )

            return stats

        return {
            "wrapper_system_type": (
                self.system_type
            ),
            "active_system": None,
            "error": "No active system",
        }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

MultimodalRAG = MultimodalRAGSystem

