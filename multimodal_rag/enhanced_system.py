
"""
Enhanced SmartRAG system.

This layer wraps SimpleRAGSystem and exposes:
- RAG question answering
- File ingestion
- File-specific summaries
- File-specific quizzes
- System status
"""

import logging
from typing import Dict, Any, Union, Optional

from .base import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


# ============================================================
# MULTIMODAL QUERY REQUEST
# ============================================================

class MultimodalQueryRequest(QueryRequest):
    """Extended multimodal query request."""

    def __init__(
        self,
        query: str = "",
        *args,
        **kwargs,
    ):
        self.search_type = kwargs.pop(
            "search_type",
            "hybrid",
        )

        self.visual_query_image = kwargs.pop(
            "visual_query_image",
            None,
        )

        if not query and args:
            query = args[0]
            args = args[1:]

        super().__init__(
            query,
            *args,
            **kwargs,
        )


# ============================================================
# ENHANCED RAG SYSTEM
# ============================================================

class EnhancedMultimodalRAGSystem:
    """
    Enhanced SmartRAG wrapper.

    The actual implementation is provided by SimpleRAGSystem.
    """

    def __init__(
        self,
        config_dict: Dict[str, Any],
    ):
        from .system import SimpleRAGSystem

        self.config = config_dict

        self._simple_system = (
            SimpleRAGSystem(
                config_dict
            )
        )

        logger.info(
            "Enhanced Multimodal RAG System initialized"
        )

    # ========================================================
    # INTERNAL COMPONENT ACCESS
    # ========================================================

    @property
    def llm(self):
        """
        Expose the underlying LLM.

        The actual LLM belongs to SimpleRAGSystem.
        """
        return getattr(
            self._simple_system,
            "llm",
            None,
        )

    @property
    def vector_store(self):
        """
        Expose the underlying ChromaDB vector store.
        """
        return getattr(
            self._simple_system,
            "vector_store",
            None,
        )

    @property
    def document_processor(self):
        return getattr(
            self._simple_system,
            "document_processor",
            None,
        )

    @property
    def image_processor(self):
        return getattr(
            self._simple_system,
            "image_processor",
            None,
        )

    @property
    def audio_processor(self):
        return getattr(
            self._simple_system,
            "audio_processor",
            None,
        )

    # ========================================================
    # STATUS
    # ========================================================

    def is_available(self) -> bool:
        """Return whether the backend is available."""

        try:
            return bool(
                self._simple_system.is_available()
            )

        except Exception as exc:
            logger.error(
                f"Availability check failed: {exc}"
            )
            return False

    # ========================================================
    # INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path,
    ):
        """Ingest a document, image, or audio file."""

        return self._simple_system.ingest_file(
            file_path
        )

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        query: Union[str, QueryRequest],
    ) -> QueryResponse:
        """Process a normal RAG query."""

        return self._simple_system.query(
            query
        )

    # ========================================================
    # FILE CHUNK RETRIEVAL
    # ========================================================

    def _get_file_chunks(
        self,
        filename: str,
        topic: str = "all",
        max_chunks: int = 16,
    ):
        """
        Retrieve chunks ONLY from the selected filename.

        This is the important fix for the problem where
        summaries kept using the first uploaded document.
        """

        if not filename:
            raise ValueError(
                "No filename was provided."
            )

        vector_store = self.vector_store

        if vector_store is None:
            raise RuntimeError(
                "Vector store is unavailable."
            )

        topic_text = (
            topic.strip()
            if topic
            else "all"
        )

        global_topics = {
            "all",
            "everything",
            "entire document",
            "whole document",
            "full document",
        }

        if topic_text.lower() in global_topics:

            search_query = (
                "main topic "
                "key concepts "
                "important facts "
                "technical details "
                "architecture "
                "process "
                "methods "
                "implementation "
                "advantages "
                "limitations "
                "challenges "
                "conclusion"
            )

        else:

            search_query = topic_text

        # ----------------------------------------------------
        # ChromaDB filter
        # ----------------------------------------------------

        retrieval_result = (
            vector_store.similarity_search(
                search_query,
                k=max_chunks,
                filter_dict={
                    "filename": filename
                },
            )
        )

        chunks = []

        for chunk in retrieval_result.chunks:

            metadata = (
                chunk.metadata
                or {}
            )

            metadata_filename = (
                metadata.get(
                    "filename"
                )
            )

            source_file = (
                chunk.source_file
                or metadata.get(
                    "source_file"
                )
            )

            source_name = ""

            if source_file:
                source_name = (
                    str(source_file)
                    .replace("\\", "/")
                    .split("/")[-1]
                )

            # Extra safety check.
            if (
                metadata_filename == filename
                or source_name == filename
            ):

                if (
                    chunk.content
                    and len(
                        chunk.content.strip()
                    ) > 20
                ):

                    chunks.append(
                        chunk
                    )

        return chunks

    # ========================================================
    # CONTEXT BUILDER
    # ========================================================

    def _build_file_context(
        self,
        chunks,
    ) -> str:
        """Build context containing source/page information."""

        context_parts = []

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
                getattr(
                    chunk,
                    "page_number",
                    None,
                )
                or metadata.get(
                    "page_number"
                )
            )

            page_info = ""

            if page_number is not None:
                page_info = (
                    f" (Page {page_number})"
                )

            context_parts.append(
                (
                    f"[Chunk {index + 1}] "
                    f"Source: {source_name}"
                    f"{page_info}\n"
                    f"{chunk.content}"
                )
            )

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # FILE-SPECIFIC SUMMARY
    # ========================================================

    def generate_summary(
        self,
        topic: str = "all",
        max_chunks: int = 8,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a summary.

        When filename is provided:
            Only that document is summarized.

        When filename is omitted:
            Falls back to the normal SimpleRAG behavior.
        """

        try:

            # ------------------------------------------------
            # Normal old behavior
            # ------------------------------------------------

            if not filename:

                return (
                    self._simple_system.generate_summary(
                        topic=topic,
                        max_chunks=max_chunks,
                    )
                )

            # ------------------------------------------------
            # File-specific behavior
            # ------------------------------------------------

            if not self.is_available():
                return "System not available."

            if self.llm is None:
                return "LLM is unavailable."

            chunks = self._get_file_chunks(
                filename=filename,
                topic=topic,
                max_chunks=max(
                    8,
                    min(
                        int(max_chunks) * 2,
                        20,
                    ),
                ),
            )

            if not chunks:

                return (
                    f"No relevant content was found "
                    f"in '{filename}'."
                )

            context = (
                self._build_file_context(
                    chunks
                )
            )

            prompt = f"""
You are EduSense AI, an offline AI study assistant.

Generate a grounded study summary ONLY from the
selected document.

SELECTED DOCUMENT:
{filename}

STRICT RULES:

1. Use ONLY the supplied document context.
2. Do NOT use information from other documents.
3. Do NOT use outside knowledge.
4. Do NOT invent facts.
5. Do NOT speculate.
6. Do NOT infer unsupported information.
7. Preserve technical terms and numerical values.
8. Do not repeat the same fact unnecessarily.
9. If a section is not supported, omit it.

Use these sections when supported:

## Topic / Overview

## Key Concepts

## Important Technical Details

## Architecture / Process

## Advantages

## Challenges / Limitations

## Important Takeaways

DOCUMENT CONTEXT:

{context}

Generate the grounded study summary now.
"""

            generation_config = (
                self.config.get(
                    "generation",
                    {},
                )
            )

            max_tokens = min(
                int(
                    generation_config.get(
                        "max_tokens",
                        512,
                    )
                ),
                768,
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

            logger.error(
                f"File-specific summary failed: {exc}"
            )

            return (
                f"Error generating summary: "
                f"{exc}"
            )

    # ========================================================
    # FILE-SPECIFIC QUIZ
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

        When filename is provided:
            Quiz uses only that document.

        When filename is omitted:
            Falls back to normal SimpleRAG behavior.
        """

        try:

            # ------------------------------------------------
            # Normal old behavior
            # ------------------------------------------------

            if not filename:

                return (
                    self._simple_system.generate_quiz(
                        topic=topic,
                        num_questions=num_questions,
                        max_chunks=max_chunks,
                    )
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if not self.is_available():
                return "System not available."

            if self.llm is None:
                return "LLM is unavailable."

            num_questions = max(
                1,
                min(
                    int(num_questions),
                    10,
                ),
            )

            chunks = self._get_file_chunks(
                filename=filename,
                topic=topic,
                max_chunks=max(
                    8,
                    min(
                        int(max_chunks) * 2,
                        20,
                    ),
                ),
            )

            if not chunks:

                return (
                    f"No relevant content was found "
                    f"in '{filename}'."
                )

            context = (
                self._build_file_context(
                    chunks
                )
            )

            prompt = f"""
You are EduSense AI, an offline AI study assistant.

Create a multiple-choice quiz ONLY from the
selected document.

SELECTED DOCUMENT:
{filename}

NUMBER OF QUESTIONS:
{num_questions}

STRICT RULES:

1. Use ONLY the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Every question must be directly answerable
   from the context.
5. Each question must have exactly four options.
6. Exactly one option must be correct.
7. The answer letter must match the correct option.
8. Give a short explanation based only on the context.
9. Do not create trick questions.
10. Do not repeat questions.
11. Preserve technical terminology.

FORMAT:

## 📝 Quiz

### Q1. Question

A. Option
B. Option
C. Option
D. Option

**Answer:** A

**Explanation:** Explanation based on the document.

DOCUMENT CONTEXT:

{context}

Generate the quiz now.
"""

            generation_config = (
                self.config.get(
                    "generation",
                    {},
                )
            )

            max_tokens = min(
                int(
                    generation_config.get(
                        "max_tokens",
                        512,
                    )
                ),
                1024,
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

            logger.error(
                f"File-specific quiz failed: {exc}"
            )

            return (
                f"Error generating quiz: "
                f"{exc}"
            )

    # ========================================================
    # STATUS
    # ========================================================

    def get_system_status(
        self,
    ) -> Dict[str, Any]:
        """Return enhanced system status."""

        try:

            status = (
                self._simple_system.get_system_status()
            )

            status["system_type"] = (
                "enhanced_traditional"
            )

            status["summary_available"] = True
            status["quiz_available"] = True

            status["llm_exposed"] = (
                self.llm is not None
            )

            status["vector_store_exposed"] = (
                self.vector_store is not None
            )

            return status

        except Exception as exc:

            logger.error(
                f"Error getting enhanced system status: {exc}"
            )

            return {
                "system_type":
                    "enhanced_traditional",

                "llm_available":
                    False,

                "summary_available":
                    False,

                "quiz_available":
                    False,

                "error":
                    str(exc),
            }

