"""
Main multimodal RAG system implementation.

SmartRAG provides:
- PDF / DOCX / TXT / Markdown ingestion
- Image and audio processing
- Ollama-based local LLM inference
- ChromaDB semantic retrieval
- RAG question answering
- Grounded document summarization
- Grounded quiz generation
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from config_schema import SmartRAGConfig, load_config

    USE_NEW_CONFIG = True

except ImportError:
    SmartRAGConfig = Any
    USE_NEW_CONFIG = False

    logging.warning(
        "config_schema not found, using legacy configuration loading"
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
# SIMPLE RAG SYSTEM
# ============================================================

class SimpleRAGSystem:
    """
    Core RAG implementation.

    Responsibilities:
    - Initialize Ollama
    - Initialize document/image/audio processors
    - Initialize ChromaDB
    - Ingest files
    - Retrieve relevant chunks
    - Generate answers
    - Generate grounded summaries
    - Generate grounded quizzes
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], SmartRAGConfig],
    ):
        """Initialize the core RAG system."""

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        if (
            USE_NEW_CONFIG
            and not isinstance(config, dict)
            and hasattr(config, "to_dict")
        ):
            self.config = config.to_dict()
            self._typed_config = config

        else:
            self.config = config
            self._typed_config = None

        logger.info(
            "Initializing Simple RAG System..."
        )

        # Safe defaults
        self.llm = None
        self.document_processor = None
        self.image_processor = None
        self.audio_processor = None
        self.vector_store = None

        try:

            # ------------------------------------------------
            # LLM
            # ------------------------------------------------

            self.llm = OllamaLLM(
                self.config
            )

            logger.info(
                "Ollama LLM initialized"
            )

            # ------------------------------------------------
            # PROCESSORS
            # ------------------------------------------------

            from .processors import (
                DocumentProcessorManager,
                ImageProcessorManager,
                AudioProcessorManager,
            )

            self.document_processor = (
                DocumentProcessorManager(
                    self.config
                )
            )

            self.image_processor = (
                ImageProcessorManager(
                    self.config
                )
            )

            self.audio_processor = (
                AudioProcessorManager(
                    self.config
                )
            )

            logger.info(
                "Document, image and audio processors initialized"
            )

            # ------------------------------------------------
            # VECTOR STORE
            # ------------------------------------------------

            from .vector_stores.chroma_store import (
                ChromaVectorStore
            )

            self.vector_store = (
                ChromaVectorStore(
                    self.config
                )
            )

            logger.info(
                "ChromaDB vector store initialized"
            )

            logger.info(
                "Simple RAG System initialized successfully"
            )

        except Exception as exc:

            logger.error(
                f"Failed to initialize Simple RAG System: {exc}"
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
        """Check whether the system is ready."""

        if self.llm is None:
            return False

        if not hasattr(
            self.llm,
            "is_available"
        ):
            return False

        try:
            return bool(
                self.llm.is_available()
            )

        except Exception as exc:

            logger.error(
                f"Error checking LLM availability: {exc}"
            )

            return False

    # ========================================================
    # FILE INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path: Union[str, Path],
    ) -> ProcessingResult:
        """
        Process and index a single file.

        Supported:
        - PDF
        - DOCX
        - TXT
        - Markdown
        - Images
        - Audio
        """

        if not self.is_available():

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message="System not available",
            )

        path = Path(file_path)

        try:

            logger.info(
                f"Ingesting file: {path}"
            )

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
            # VECTOR STORE
            # ------------------------------------------------

            if (
                result.success
                and result.chunks
                and self.vector_store
            ):

                stored = (
                    self.vector_store.add_documents(
                        result.chunks
                    )
                )

                if stored is False:

                    logger.warning(
                        "Vector store did not confirm "
                        "document insertion."
                    )

                logger.info(
                    f"Added {len(result.chunks)} "
                    f"chunks to vector store"
                )

            return result

        except Exception as exc:

            logger.error(
                f"Error ingesting file {path}: {exc}"
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
        """
        Process a user question using RAG.
        """

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

            if isinstance(
                query,
                str
            ):

                query_text = query

                query_obj = QueryRequest(
                    query=query
                )

            else:

                query_text = query.query

                query_obj = query

            logger.info(
                f"Processing query: {query_text}"
            )

            # ------------------------------------------------
            # CONVERSATIONAL DETECTION
            # ------------------------------------------------

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
                        {}
                    )
                    .get(
                        "top_k",
                        5
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
                        8
                    )
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
                        query_text,
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
                    f"Retrieved "
                    f"{len(relevant_chunks)} "
                    f"relevant chunks"
                )

            else:

                logger.info(
                    "Treating query as conversational"
                )

            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            response_text = (
                self._generate_contextual_response(
                    query_text,
                    context,
                    is_conversational,
                    **query_obj.generation_params,
                )
            )

            confidence = (
                0.8
                if sources
                else 0.9
            )

            return QueryResponse(
                answer=response_text,
                sources=sources,
                query=query_text,
                confidence_score=confidence,
            )

        except Exception as exc:

            logger.error(
                f"Error processing query: {exc}"
            )

            return QueryResponse(
                answer=(
                    f"Error processing query: "
                    f"{str(exc)}"
                ),
                sources=[],
                query=query_text,
                confidence_score=0.0,
            )

    # ========================================================
    # SUMMARY GENERATION
    # ========================================================

    def generate_summary(
        self,
        topic: str = "all",
        max_chunks: int = 8,
    ) -> str:
        """
        Generate a grounded study summary.
        """

        if not self.is_available():
            return "System not available."

        if not self.vector_store:
            return (
                "Vector store is not available."
            )

        try:

            # ------------------------------------------------
            # NORMALIZE TOPIC
            # ------------------------------------------------

            topic_text = (
                topic.strip()
                if topic
                else "all"
            )

            is_global_summary = (
                topic_text.lower()
                in {
                    "all",
                    "everything",
                    "entire document",
                    "whole document",
                    "full document",
                }
            )

            # ------------------------------------------------
            # SEARCH QUERY
            # ------------------------------------------------

            if is_global_summary:

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

                search_query = topic_text

            logger.info(
                f"Generating summary for: "
                f"{search_query}"
            )

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            candidate_count = min(
                max(
                    int(max_chunks) * 2,
                    12
                ),
                20
            )

            retrieval_result = (
                self.vector_store.similarity_search(
                    search_query,
                    k=candidate_count,
                )
            )

            # ------------------------------------------------
            # CHUNK SELECTION
            # ------------------------------------------------

            if is_global_summary:

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
                        search_query,
                        max_chunks=max_chunks,
                    )
                )

            if not relevant_chunks:

                return (
                    "No relevant document content "
                    "was found for summarization."
                )

            logger.info(
                f"Selected "
                f"{len(relevant_chunks)} "
                f"chunks for summary"
            )

            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            context = (
                self._build_context(
                    relevant_chunks
                )
            )

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------

            prompt = f"""
You are EduSense AI, an offline AI study assistant.

Create a factually grounded study summary from
the supplied document context.

CRITICAL RULES:

1. Use ONLY information explicitly contained
   in the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT speculate.
4. Do NOT infer missing information.
5. Do NOT invent facts.
6. If a section is unsupported, OMIT it.
7. Preserve names, technical terms and numbers.
8. Keep the summary useful for student revision.
9. Prefer concise bullet points.
10. Do not repeat the same fact unnecessarily.

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

Generate the grounded study summary.
"""

            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            generation_config = (
                self.config.get(
                    "generation",
                    {}
                )
            )

            configured_max_tokens = (
                generation_config.get(
                    "max_tokens",
                    512
                )
            )

            summary_max_tokens = min(
                int(configured_max_tokens),
                768
            )

            summary = (
                self.llm.generate_response(
                    prompt=prompt,
                    context="",
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=summary_max_tokens,
                )
            )

            return summary.strip()

        except Exception as exc:

            logger.error(
                f"Error generating summary: {exc}"
            )

            return (
                f"Error generating summary: "
                f"{str(exc)}"
            )

    # ========================================================
    # QUIZ GENERATION
    # ========================================================

    def generate_quiz(
        self,
        topic: str = "all",
        num_questions: int = 5,
        max_chunks: int = 8,
    ) -> str:
        """
        Generate a grounded multiple-choice quiz
        from indexed documents.
        """

        if not self.is_available():
            return "System not available."

        if not self.vector_store:
            return (
                "Vector store is not available."
            )

        try:

            # ------------------------------------------------
            # NORMALIZE PARAMETERS
            # ------------------------------------------------

            num_questions = max(
                1,
                min(
                    int(num_questions),
                    10
                )
            )

            max_chunks = max(
                1,
                min(
                    int(max_chunks),
                    10
                )
            )

            topic_text = (
                topic.strip()
                if topic
                else "all"
            )

            is_global_quiz = (
                topic_text.lower()
                in {
                    "all",
                    "everything",
                    "entire document",
                    "whole document",
                    "full document",
                }
            )

            # ------------------------------------------------
            # SEARCH QUERY
            # ------------------------------------------------

            if is_global_quiz:

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

                search_query = topic_text

            logger.info(
                f"Generating quiz for: "
                f"{search_query}"
            )

            # ------------------------------------------------
            # RETRIEVE CANDIDATES
            # ------------------------------------------------

            candidate_count = min(
                max(
                    max_chunks * 2,
                    12
                ),
                20
            )

            retrieval_result = (
                self.vector_store.similarity_search(
                    search_query,
                    k=candidate_count,
                )
            )

            # ------------------------------------------------
            # SELECT DIVERSE CHUNKS
            # ------------------------------------------------

            if is_global_quiz:

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
                        search_query,
                        max_chunks=max_chunks,
                    )
                )

            if not relevant_chunks:

                return (
                    "No relevant document content "
                    "was found for quiz generation."
                )

            context = (
                self._build_context(
                    relevant_chunks
                )
            )

            # ------------------------------------------------
            # QUIZ PROMPT
            # ------------------------------------------------

            prompt = f"""
You are EduSense AI, an offline AI study assistant.

Create a grounded multiple-choice quiz using ONLY
information explicitly contained in the document context.

Number of questions:
{num_questions}

STRICT RULES:

1. Every question MUST be answerable directly from
   the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Each question must contain exactly four options:
   A, B, C and D.
5. Exactly ONE option must be correct.
6. The correct answer letter MUST correspond exactly
   to the option containing the correct information.
7. The explanation MUST support that same option.
8. Before writing the final answer, internally verify:
   - the question is supported by the context
   - the correct option is correct
   - the Answer letter matches that option
   - the Explanation matches that option
9. Do not create trick questions.
10. Do not repeat questions.
11. Preserve technical terminology exactly.
12. If the context does not contain enough information
    for a reliable question, do not create that question.

Format:

## 📝 Quiz

### Q1. Question

A. Option
B. Option
C. Option
D. Option

**Answer:** A

**Explanation:** Explanation supported directly by the document.

Generate {num_questions} questions.

DOCUMENT CONTEXT:

{context}
"""

            # ------------------------------------------------
            # SAFE GENERATION
            # ------------------------------------------------

            generation_config = (
                self.config.get(
                    "generation",
                    {}
                )
            )

            configured_max_tokens = (
                generation_config.get(
                    "max_tokens",
                    512
                )
            )

            quiz_max_tokens = min(
                int(configured_max_tokens),
                1024
            )

            quiz = (
                self.llm.generate_response(
                    prompt=prompt,
                    context="",
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=quiz_max_tokens,
                )
            )

            return quiz.strip()

        except Exception as exc:

            logger.error(
                f"Error generating quiz: {exc}"
            )

            return (
                f"Error generating quiz: "
                f"{str(exc)}"
            )

    # ========================================================
    # SUMMARY CHUNK SELECTION
    # ========================================================

    def _select_summary_chunks(
        self,
        chunks: List[DocumentChunk],
        max_chunks: int = 8,
    ) -> List[DocumentChunk]:
        """
        Select diverse chunks for summaries and quizzes.

        Strategy:
        1. Remove empty chunks.
        2. Prefer different pages.
        3. Fill remaining slots with ranked chunks.
        """

        if not chunks:
            return []

        valid_chunks = [
            chunk
            for chunk in chunks
            if (
                chunk.content
                and len(
                    chunk.content.strip()
                ) > 50
            )
        ]

        if not valid_chunks:
            return []

        max_chunks = max(
            1,
            min(
                int(max_chunks),
                len(valid_chunks)
            )
        )

        selected: List[
            DocumentChunk
        ] = []

        selected_ids = set()

        seen_pages = set()

        # ----------------------------------------------------
        # PASS 1
        # Prefer different pages.
        # ----------------------------------------------------

        for chunk in valid_chunks:

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

            selected.append(chunk)

            selected_ids.add(
                chunk.chunk_id
            )

            seen_pages.add(
                page_number
            )

            if len(selected) >= max_chunks:
                return selected

        # ----------------------------------------------------
        # PASS 2
        # Fill remaining positions.
        # ----------------------------------------------------

        for chunk in valid_chunks:

            if chunk.chunk_id in selected_ids:
                continue

            selected.append(chunk)

            selected_ids.add(
                chunk.chunk_id
            )

            if len(selected) >= max_chunks:
                break

        return selected

    # ========================================================
    # NORMAL CONTEXT FILTERING
    # ========================================================

    def _filter_relevant_context(
        self,
        chunks: List[DocumentChunk],
        query_text: str,
        min_score: float = 0.1,
        max_chunks: int = 5,
    ) -> List[DocumentChunk]:
        """
        Lightweight relevance filtering.

        ChromaDB already performs semantic retrieval.
        This method removes only unusable chunks.
        """

        if not chunks:
            return []

        valid_chunks = [
            chunk
            for chunk in chunks
            if (
                chunk.content
                and len(
                    chunk.content.strip()
                ) > 50
            )
        ]

        return valid_chunks[
            :max_chunks
        ]

    # ========================================================
    # CONVERSATIONAL QUERY DETECTION
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
            query_text
            .lower()
            .strip()
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
    # CONTEXT BUILDING
    # ========================================================

    def _build_context(
        self,
        chunks: List[DocumentChunk],
    ) -> str:
        """Build context with source/page information."""

        context_parts = []

        for index, chunk in enumerate(
            chunks
        ):

            source_file = (
                chunk.source_file
                or chunk.metadata.get(
                    "filename",
                    "Unknown Document",
                )
            )

            source_name = (
                str(source_file)
                .replace("\\", "/")
                .split("/")[-1]
            )

            if not source_name:
                source_name = (
                    "Unknown Document"
                )

            page_number = (
                chunk.page_number
                or chunk.metadata.get(
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
    # NORMAL RESPONSE GENERATION
    # ========================================================

    def _generate_contextual_response(
        self,
        query_text: str,
        context: str,
        is_conversational: bool,
        **kwargs,
    ) -> str:
        """Generate normal RAG response."""

        if is_conversational:

            full_prompt = f"""
You are a helpful offline AI assistant.

Respond naturally and briefly.

User:
{query_text}

Assistant:
"""

        else:

            if context.strip():

                full_prompt = f"""
You are a helpful offline AI study assistant.

Answer the user's question using the supplied
document context.

Rules:
- Use the context as the primary source.
- Do not invent document-specific facts.
- Mention the source document when appropriate.
- If the answer is not contained in the context,
  say that it was not found.
- Do not pretend outside knowledge came from the
  uploaded document.

DOCUMENT CONTEXT:

{context}

QUESTION:

{query_text}

ANSWER:
"""

            else:

                full_prompt = f"""
You are a helpful offline AI assistant.

No relevant document context was retrieved.

Answer the question generally, but do not claim
that the answer came from the user's files.

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
                768
            )

        return (
            self.llm.generate_response(
                prompt=full_prompt,
                context="",
                **safe_kwargs,
            )
        )

    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    def get_system_status(
        self,
    ) -> Dict[str, Any]:
        """Return system health."""

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
                    .get("models", {})
                    .get(
                        "llm_model",
                        "unknown",
                    ),

                "embedding_model":
                    self.config
                    .get("models", {})
                    .get(
                        "embedding_model",
                        "unknown",
                    ),
            }

        except Exception as exc:

            logger.error(
                f"Error getting system status: {exc}"
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

    Uses:
        EnhancedMultimodalRAGSystem
    with:
        SimpleRAGSystem fallback.
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
        """Initialize SmartRAG."""

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        if USE_NEW_CONFIG:

            try:

                typed_config = (
                    load_config(
                        config_path=config_path,
                        **overrides,
                    )
                )

                config = (
                    typed_config.to_dict()
                )

                logger.info(
                    "Using validated configuration schema"
                )

            except Exception as exc:

                logger.warning(
                    "Validated configuration failed: "
                    f"{exc}"
                )

                config = (
                    self._load_config_legacy(
                        config_path,
                        config_dict,
                    )
                )

        else:

            config = (
                self._load_config_legacy(
                    config_path,
                    config_dict,
                )
            )

        self.config = config

        # ----------------------------------------------------
        # BACKEND
        # ----------------------------------------------------

        self._system = None
        self.system_type = "none"

        # Try enhanced system.
        try:

            from .enhanced_system import (
                EnhancedMultimodalRAGSystem
            )

            enhanced_system = (
                EnhancedMultimodalRAGSystem(
                    config
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
                    "Enhanced RAG system initialized successfully"
                )

        except Exception as exc:

            logger.warning(
                f"Enhanced system initialization failed: "
                f"{exc}"
            )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if self._system is None:

            try:

                fallback_system = (
                    SimpleRAGSystem(
                        config
                    )
                )

                if fallback_system.is_available():

                    self._system = (
                        fallback_system
                    )

                    self.system_type = (
                        "simple_traditional"
                    )

                    logger.info(
                        "Simple RAG fallback initialized"
                    )

                else:

                    logger.error(
                        "Simple RAG fallback unavailable"
                    )

            except Exception as exc:

                logger.error(
                    f"Failed to initialize fallback: "
                    f"{exc}"
                )

    # ========================================================
    # LEGACY CONFIGURATION
    # ========================================================

    def _load_config_legacy(
        self,
        config_path: Optional[
            Union[str, Path]
        ],
        config_dict: Optional[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """Load legacy configuration."""

        if config_dict:
            return config_dict

        if config_path:
            return self._load_config(
                config_path
            )

        return self._get_default_config()

    def _load_config(
        self,
        config_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load YAML configuration."""

        import yaml

        try:

            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as file:

                config = yaml.safe_load(
                    file
                )

            return config

        except Exception as exc:

            logger.error(
                f"Failed to load config: {exc}"
            )

            return (
                self._get_default_config()
            )

    def _get_default_config(
        self,
    ) -> Dict[str, Any]:
        """Default local configuration."""

        return {
            "system": {
                "name": "SmartRAG System",
                "offline_mode": True,
                "debug": False,
                "log_level": "INFO",
            },

            "models": {
                "llm_type": "ollama",

                "llm_model":
                    "qwen2.5:3b",

                "ollama_host":
                    "http://localhost:11434",

                "embedding_model":
                    "nomic-embed-text",

                "embedding_dimension":
                    768,

                "vision_model":
                    "Salesforce/"
                    "blip-image-captioning-base",

                "whisper_model":
                    "base",

                "whisper_device":
                    "cpu",
            },

            "vector_store": {
                "type":
                    "chromadb",

                "persist_directory":
                    "./vector_db",

                "collection_name":
                    "multimodal_documents",

                "embedding_dimension":
                    768,

                "ollama_host":
                    "http://localhost:11434",
            },

            "processing": {
                "chunk_size":
                    1000,

                "chunk_overlap":
                    200,

                "max_image_size":
                    [1024, 1024],

                "ocr_enabled":
                    True,

                "batch_size":
                    16,

                "store_original_images":
                    True,

                "image_preprocessing":
                    "resize",

                "audio_sample_rate":
                    16000,

                "max_audio_duration":
                    300,
            },

            "retrieval": {
                "top_k":
                    5,

                "similarity_threshold":
                    0.7,

                "rerank_enabled":
                    False,
            },

            "generation": {
                "max_tokens":
                    512,

                "temperature":
                    0.7,

                "top_p":
                    0.9,

                "top_k":
                    50,

                "do_sample":
                    True,

                "max_new_tokens":
                    512,
            },
        }

    # ========================================================
    # PUBLIC INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path: Union[str, Path],
    ) -> ProcessingResult:
        """Ingest a file."""

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
    # PUBLIC STATUS
    # ========================================================

    def is_available(self) -> bool:
        """Check system availability."""

        return (
            self._system is not None
            and self._system.is_available()
        )

    # ========================================================
    # PUBLIC QUERY
    # ========================================================

    def query(
        self,
        query: Union[str, QueryRequest],
    ) -> QueryResponse:
        """Send query to RAG system."""

        if self._system is None:

            query_text = (
                query
                if isinstance(query, str)
                else query.query
            )

            return QueryResponse(
                answer=(
                    "No active RAG system. "
                    "Please check Ollama and "
                    "the configured local model."
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
    # PUBLIC SUMMARY
    # ========================================================

    def generate_summary(
        self,
        topic: str = "all",
        max_chunks: int = 8,
    ) -> str:
        """Generate grounded summary."""

        if self._system is None:

            return (
                "No active RAG system."
            )

        if not hasattr(
            self._system,
            "generate_summary",
        ):

            return (
                "Summary generation is not "
                "available in the active backend."
            )

        return (
            self._system.generate_summary(
                topic=topic,
                max_chunks=max_chunks,
            )
        )

    # ========================================================
    # PUBLIC QUIZ
    # ========================================================

    def generate_quiz(
        self,
        topic: str = "all",
        num_questions: int = 5,
        max_chunks: int = 8,
    ) -> str:
        """Generate grounded multiple-choice quiz."""

        if self._system is None:

            return (
                "No active RAG system."
            )

        if not hasattr(
            self._system,
            "generate_quiz",
        ):

            return (
                "Quiz generation is not "
                "available in the active backend."
            )

        return (
            self._system.generate_quiz(
                topic=topic,
                num_questions=num_questions,
                max_chunks=max_chunks,
            )
        )

    # ========================================================
    # SYSTEM STATISTICS
    # ========================================================

    def get_system_stats(
        self,
    ) -> Dict[str, Any]:
        """Return system statistics."""

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

            return stats

        return {
            "wrapper_system_type":
                self.system_type,

            "active_system":
                None,

            "error":
                "No active system",
        }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

MultimodalRAG = MultimodalRAGSystem