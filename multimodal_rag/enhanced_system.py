"""
Enhanced SmartRAG system.

This layer delegates core functionality to SimpleRAGSystem
while exposing the complete public API.
"""

import logging
from typing import Dict, Any, Union

from .base import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


class MultimodalQueryRequest(QueryRequest):
    """Extended multimodal query request."""

    def __init__(
        self,
        query: str = "",
        *args,
        **kwargs
    ):
        self.search_type = kwargs.pop(
            "search_type",
            "hybrid"
        )

        self.visual_query_image = kwargs.pop(
            "visual_query_image",
            None
        )

        if not query and args:
            query = args[0]
            args = args[1:]

        super().__init__(
            query,
            *args,
            **kwargs
        )


class EnhancedMultimodalRAGSystem:
    """
    Enhanced RAG wrapper.

    Core implementation is provided by SimpleRAGSystem.
    """

    def __init__(
        self,
        config_dict: Dict[str, Any]
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
    # STATUS
    # ========================================================

    def is_available(self) -> bool:
        """Return whether the backend is available."""

        return (
            self._simple_system.is_available()
        )

    # ========================================================
    # INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path
    ):
        """Ingest a document, image or audio file."""

        return (
            self._simple_system.ingest_file(
                file_path
            )
        )

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        query: Union[str, QueryRequest]
    ) -> QueryResponse:
        """Process a RAG query."""

        return (
            self._simple_system.query(
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
    ) -> str:
        """Generate a grounded study summary."""

        return (
            self._simple_system.generate_summary(
                topic=topic,
                max_chunks=max_chunks,
            )
        )

    # ========================================================
    # QUIZ
    # ========================================================

    def generate_quiz(
        self,
        topic: str = "all",
        num_questions: int = 5,
        max_chunks: int = 8,
    ) -> str:
        """Generate a grounded multiple-choice quiz."""

        return (
            self._simple_system.generate_quiz(
                topic=topic,
                num_questions=num_questions,
                max_chunks=max_chunks,
            )
        )

    # ========================================================
    # STATUS INFORMATION
    # ========================================================

    def get_system_status(
        self
    ) -> Dict[str, Any]:
        """Return backend status."""

        status = (
            self._simple_system.get_system_status()
        )

        status["system_type"] = (
            "enhanced_traditional"
        )

        status["summary_available"] = True
        status["quiz_available"] = True

        return status