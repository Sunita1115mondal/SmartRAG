
"""
ChromaDB vector store implementation for SmartRAG.

Features:
- Persistent ChromaDB storage
- Ollama embeddings using nomic-embed-text
- Semantic similarity search
- Metadata filtering by filename/source
- Document-specific retrieval
- Collection statistics
- Safe document insertion and deletion
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    import ollama
    import requests
except ImportError as exc:
    chromadb = None
    ollama = None
    requests = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


from ..base import BaseVectorStore, DocumentChunk, RetrievalResult


logger = logging.getLogger(__name__)


# ============================================================
# OLLAMA EMBEDDING FUNCTION
# ============================================================

class OllamaEmbeddingFunction:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")

        if ollama is None or requests is None:
            raise ImportError(
                "Required packages are missing. "
                "Install with: pip install ollama requests"
            )

        self._check_ollama_connection()
        self._ensure_model_available()

    def name(self) -> str:
        """Return a stable embedding-function name."""
        return f"ollama-{self.model_name}"

    def _check_ollama_connection(self) -> None:
        """Verify that Ollama is running."""
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5,
            )

            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama returned HTTP {response.status_code}"
                )

            logger.info(
                "Ollama server available at %s",
                self.host,
            )

        except requests.exceptions.RequestException as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                f"Make sure Ollama is running. Error: {exc}"
            ) from exc

    def _ensure_model_available(self) -> None:
        """Ensure the embedding model is available."""
        try:
            response = ollama.list()

            model_names = []

            for model in getattr(response, "models", []):
                model_name = getattr(model, "model", None)

                if model_name:
                    model_names.append(model_name)

            logger.debug(
                "Available Ollama models: %s",
                model_names,
            )

            # Exact match
            if self.model_name in model_names:
                logger.info(
                    "Embedding model available: %s",
                    self.model_name,
                )
                return

            # :latest match
            latest_name = f"{self.model_name}:latest"

            if latest_name in model_names:
                self.model_name = latest_name
                logger.info(
                    "Using embedding model: %s",
                    self.model_name,
                )
                return

            # Partial match
            matching = [
                name
                for name in model_names
                if self.model_name in name
            ]

            if matching:
                self.model_name = matching[0]
                logger.info(
                    "Using matching embedding model: %s",
                    self.model_name,
                )
                return

            # Pull model if missing
            logger.info(
                "Embedding model '%s' not found. Pulling...",
                self.model_name,
            )

            ollama.pull(self.model_name)

            logger.info(
                "Successfully pulled embedding model: %s",
                self.model_name,
            )

        except Exception as exc:
            logger.error(
                "Failed to prepare embedding model '%s': %s",
                self.model_name,
                exc,
            )
            raise

    def __call__(
        self,
        input: List[str],
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not input:
            return []

        try:
            embeddings: List[List[float]] = []

            for text in input:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=str(text),
                )

                embedding = response.get("embedding")

                if not embedding:
                    raise ValueError(
                        "Ollama returned an empty embedding."
                    )

                embeddings.append(embedding)

            logger.debug(
                "Generated %d embeddings using %s",
                len(embeddings),
                self.model_name,
            )

            return embeddings

        except Exception as exc:
            logger.error(
                "Error generating embeddings: %s",
                exc,
            )
            raise

    def embed_query(
        self,
        *args,
        **kwargs,
    ) -> List[float]:
        """Generate one embedding for a query."""
        try:
            query = None

            if args:
                query = args[0]
            elif "query" in kwargs:
                query = kwargs["query"]
            elif "input" in kwargs:
                query = kwargs["input"]

            if isinstance(query, list):
                query = query[0] if query else ""

            if query is None:
                raise ValueError(
                    "No query provided for embedding."
                )

            response = ollama.embeddings(
                model=self.model_name,
                prompt=str(query),
            )

            embedding = response.get("embedding")

            if not embedding:
                raise ValueError(
                    "Ollama returned an empty query embedding."
                )

            return embedding

        except Exception as exc:
            logger.error(
                "Error generating query embedding: %s",
                exc,
            )
            raise


# ============================================================
# CHROMA VECTOR STORE
# ============================================================

class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector store."""

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config)

        if _IMPORT_ERROR is not None:
            raise ImportError(
                "ChromaDB/Ollama dependencies are unavailable."
            ) from _IMPORT_ERROR

        if chromadb is None:
            raise ImportError(
                "ChromaDB is not installed."
            )

        if ollama is None:
            raise ImportError(
                "Ollama Python package is not installed."
            )

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        self.persist_directory = (
            config.get(
                "persist_directory",
                "./vector_db",
            )
        )

        self.embedding_function_name = (
            config.get(
                "embedding_model",
                "nomic-embed-text",
            )
        )

        self.ollama_host = (
            config.get(
                "ollama_host",
                "http://localhost:11434",
            )
        )

        self.client = None
        self.collection = None
        self.embedding_function = None

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        self._initialize_client()
        self._initialize_collection()

    # ========================================================
    # CLIENT
    # ========================================================

    def _initialize_client(self) -> None:
        """Initialize persistent ChromaDB."""
        try:
            settings = Settings(
                anonymized_telemetry=False,
            )

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings,
            )

            logger.info(
                "ChromaDB initialized at: %s",
                self.persist_directory,
            )

        except Exception as exc:
            logger.error(
                "Failed to initialize ChromaDB: %s",
                exc,
            )
            raise

    # ========================================================
    # COLLECTION
    # ========================================================

    def _initialize_collection(self) -> None:
        """Load or create the configured collection."""
        try:
            self.embedding_function = (
                OllamaEmbeddingFunction(
                    model_name=self.embedding_function_name,
                    host=self.ollama_host,
                )
            )

            logger.info(
                "Using Ollama embedding model: %s",
                self.embedding_function.model_name,
            )

            # BaseVectorStore normally provides collection_name.
            # Keep "default" as the fallback because that is the
            # collection currently used by your project.
            collection_name = getattr(
                self,
                "collection_name",
                None,
            ) or self.config.get(
                "collection_name",
                "default",
            )

            self.collection_name = collection_name

            try:
                self.collection = (
                    self.client.get_collection(
                        name=self.collection_name,
                        embedding_function=(
                            self.embedding_function
                        ),
                    )
                )

                logger.info(
                    "Loaded existing ChromaDB collection: %s",
                    self.collection_name,
                )

            except Exception:

                self.collection = (
                    self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=(
                            self.embedding_function
                        ),
                        metadata={
                            "hnsw:space": "cosine"
                        },
                    )
                )

                logger.info(
                    "Created ChromaDB collection: %s",
                    self.collection_name,
                )

        except Exception as exc:
            logger.error(
                "Failed to initialize ChromaDB collection: %s",
                exc,
            )
            raise

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        chunks: List[DocumentChunk],
    ) -> bool:
        """Insert document chunks into ChromaDB."""

        if not chunks:
            return True

        try:
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for chunk in chunks:

                chunk_id = (
                    chunk.chunk_id
                    or str(uuid.uuid4())
                )

                ids.append(chunk_id)
                documents.append(
                    chunk.content or ""
                )

                metadata = self._prepare_metadata(
                    chunk.metadata or {}
                )

                # ------------------------------------------------
                # STANDARD METADATA
                # ------------------------------------------------

                source_file = (
                    chunk.source_file
                    or metadata.get(
                        "source_file",
                        "unknown",
                    )
                )

                filename = metadata.get(
                    "filename"
                )

                # Make sure filename is always present.
                if not filename and source_file:
                    filename = (
                        str(source_file)
                        .replace("\\", "/")
                        .split("/")[-1]
                    )

                metadata.update(
                    {
                        "filename": str(
                            filename or "unknown"
                        ),
                        "source_file": str(
                            source_file or "unknown"
                        ),
                        "document_type": str(
                            chunk.document_type
                            or metadata.get(
                                "document_type",
                                "unknown",
                            )
                        ),
                    }
                )

                if chunk.timestamp:
                    metadata["timestamp"] = (
                        chunk.timestamp.isoformat()
                    )

                # Chroma metadata cannot contain None.
                metadata = {
                    key: value
                    for key, value in metadata.items()
                    if value is not None
                }

                metadatas.append(metadata)

            # ------------------------------------------------
            # UPSERT INSTEAD OF ADD
            # ------------------------------------------------
            #
            # This is important.
            #
            # If the same chunk ID already exists, add()
            # can fail. upsert() safely replaces that chunk.
            #
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "Upserted %d chunks into collection '%s'",
                len(chunks),
                self.collection_name,
            )

            return True

        except Exception as exc:
            logger.error(
                "Failed to add documents to ChromaDB: %s",
                exc,
            )
            return False

    # ========================================================
    # SIMILARITY SEARCH
    # ========================================================

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[
            Dict[str, Any]
        ] = None,
    ) -> RetrievalResult:
        """
        Perform semantic similarity search.

        filter_dict allows document-specific retrieval.

        Examples:

            similarity_search(
                "main topic",
                k=5,
                filter_dict={
                    "filename": "MinalRanjit-2.docx"
                },
            )

        This means ChromaDB searches ONLY that file.
        """

        start_time = time.time()

        try:

            # ------------------------------------------------
            # VALIDATE K
            # ------------------------------------------------

            k = max(
                1,
                int(k),
            )

            # ------------------------------------------------
            # BUILD WHERE FILTER
            # ------------------------------------------------

            where_clause = None

            if filter_dict:
                where_clause = (
                    self._prepare_where_clause(
                        filter_dict
                    )
                )

                logger.info(
                    "Using ChromaDB metadata filter: %s",
                    where_clause,
                )

            # ------------------------------------------------
            # DETERMINE AVAILABLE RESULTS
            # ------------------------------------------------

            available_count = (
                self.collection.count()
            )

            if available_count <= 0:

                return RetrievalResult(
                    chunks=[],
                    scores=[],
                    query=query,
                    total_results=0,
                    retrieval_time=(
                        time.time()
                        - start_time
                    ),
                )

            # ------------------------------------------------
            # QUERY WITH EMBEDDING
            # ------------------------------------------------

            query_embedding = (
                self.embedding_function.embed_query(
                    query
                )
            )

            # ------------------------------------------------
            # SAFELY DETERMINE N_RESULTS
            # ------------------------------------------------

            n_results = min(
                k,
                available_count,
            )

            # When a filter is used, Chroma can return fewer
            # records than the entire collection. We deliberately
            # keep n_results bounded and let Chroma apply the filter.
            results = self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=n_results,
                where=where_clause,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            # ------------------------------------------------
            # CONVERT RESULTS
            # ------------------------------------------------

            chunks: List[DocumentChunk] = []
            scores: List[float] = []

            documents = (
                results.get("documents") or [[]]
            )

            metadatas = (
                results.get("metadatas") or [[]]
            )

            distances = (
                results.get("distances") or [[]]
            )

            ids = (
                results.get("ids") or [[]]
            )

            result_documents = (
                documents[0]
                if documents
                else []
            )

            result_metadatas = (
                metadatas[0]
                if metadatas
                else []
            )

            result_distances = (
                distances[0]
                if distances
                else []
            )

            result_ids = (
                ids[0]
                if ids
                else []
            )

            for index, document in enumerate(
                result_documents
            ):

                metadata = (
                    result_metadatas[index]
                    if index < len(
                        result_metadatas
                    )
                    else {}
                )

                distance = (
                    result_distances[index]
                    if index < len(
                        result_distances
                    )
                    else 1.0
                )

                chunk_id = (
                    result_ids[index]
                    if index < len(
                        result_ids
                    )
                    else str(uuid.uuid4())
                )

                # Cosine distance → similarity.
                similarity_score = (
                    1.0 - float(distance)
                )

                scores.append(
                    similarity_score
                )

                chunk = DocumentChunk(
                    content=document,
                    metadata=metadata,
                    document_type=metadata.get(
                        "document_type",
                        "unknown",
                    ),
                    chunk_id=chunk_id,
                    source_file=metadata.get(
                        "source_file",
                        metadata.get(
                            "filename",
                            "unknown",
                        ),
                    ),
                )

                chunks.append(chunk)

            retrieval_time = (
                time.time()
                - start_time
            )

            logger.info(
                "Similarity search returned %d chunks",
                len(chunks),
            )

            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                total_results=len(chunks),
                retrieval_time=retrieval_time,
            )

        except Exception as exc:

            logger.error(
                "Similarity search failed: %s",
                exc,
            )

            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                total_results=0,
                retrieval_time=(
                    time.time()
                    - start_time
                ),
            )

    # ========================================================
    # DOCUMENT-SPECIFIC SEARCH
    # ========================================================

    def similarity_search_by_filename(
        self,
        query: str,
        filename: str,
        k: int = 5,
    ) -> RetrievalResult:
        """
        Search only inside one uploaded document.

        This is the method the Study Tools should use.
        """

        clean_filename = (
            str(filename)
            .replace("\\", "/")
            .split("/")[-1]
            .strip()
        )

        if not clean_filename:
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                total_results=0,
                retrieval_time=0.0,
            )

        logger.info(
            "Document-specific search: %s",
            clean_filename,
        )

        return self.similarity_search(
            query=query,
            k=k,
            filter_dict={
                "filename": clean_filename,
            },
        )

    # ========================================================
    # GET FILE CHUNKS
    # ========================================================

    def get_chunks_by_filename(
        self,
        filename: str,
    ) -> List[DocumentChunk]:
        """
        Retrieve all stored chunks for a specific filename.

        This is useful for whole-document summaries.
        """

        start_time = time.time()

        try:

            clean_filename = (
                str(filename)
                .replace("\\", "/")
                .split("/")[-1]
                .strip()
            )

            if not clean_filename:
                return []

            results = self.collection.get(
                where={
                    "filename": clean_filename
                },
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            documents = (
                results.get("documents") or []
            )

            metadatas = (
                results.get("metadatas") or []
            )

            ids = (
                results.get("ids") or []
            )

            chunks: List[DocumentChunk] = []

            for index, document in enumerate(
                documents
            ):

                metadata = (
                    metadatas[index]
                    if index < len(
                        metadatas
                    )
                    else {}
                )

                chunk_id = (
                    ids[index]
                    if index < len(ids)
                    else str(uuid.uuid4())
                )

                chunks.append(
                    DocumentChunk(
                        content=document,
                        metadata=metadata,
                        document_type=metadata.get(
                            "document_type",
                            "unknown",
                        ),
                        chunk_id=chunk_id,
                        source_file=metadata.get(
                            "source_file",
                            metadata.get(
                                "filename",
                                clean_filename,
                            ),
                        ),
                    )
                )

            # Keep original chunk order when possible.
            chunks.sort(
                key=lambda chunk: int(
                    chunk.metadata.get(
                        "chunk_index",
                        0,
                    )
                )
            )

            logger.info(
                "Retrieved %d chunks for '%s' in %.3fs",
                len(chunks),
                clean_filename,
                time.time() - start_time,
            )

            return chunks

        except Exception as exc:

            logger.error(
                "Failed to get chunks for '%s': %s",
                filename,
                exc,
            )

            return []

    # ========================================================
    # LIST DOCUMENTS
    # ========================================================

    def list_filenames(self) -> List[str]:
        """Return unique filenames stored in ChromaDB."""

        try:

            data = self.collection.get(
                include=["metadatas"]
            )

            metadatas = (
                data.get("metadatas") or []
            )

            filenames = set()

            for metadata in metadatas:

                filename = metadata.get(
                    "filename"
                )

                if filename:
                    filenames.add(
                        str(filename)
                    )

            return sorted(
                filenames,
                key=str.lower,
            )

        except Exception as exc:

            logger.error(
                "Failed to list filenames: %s",
                exc,
            )

            return []

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_by_filename(
        self,
        filename: str,
    ) -> bool:
        """Delete all vector chunks belonging to one file."""

        try:

            clean_filename = (
                str(filename)
                .replace("\\", "/")
                .split("/")[-1]
                .strip()
            )

            if not clean_filename:
                return False

            results = self.collection.get(
                where={
                    "filename": clean_filename
                },
                include=[],
            )

            ids = (
                results.get("ids") or []
            )

            if ids:
                self.collection.delete(
                    ids=ids
                )

                logger.info(
                    "Deleted %d chunks belonging to '%s'",
                    len(ids),
                    clean_filename,
                )

            return True

        except Exception as exc:

            logger.error(
                "Failed to delete document '%s': %s",
                filename,
                exc,
            )

            return False

    # ========================================================
    # DELETE BY CHUNK IDs
    # ========================================================

    def delete_documents(
        self,
        chunk_ids: List[str],
    ) -> bool:
        """Delete specific chunks by ID."""

        if not chunk_ids:
            return True

        try:

            self.collection.delete(
                ids=chunk_ids
            )

            logger.info(
                "Deleted %d chunks",
                len(chunk_ids),
            )

            return True

        except Exception as exc:

            logger.error(
                "Failed to delete chunks: %s",
                exc,
            )

            return False

    # ========================================================
    # COLLECTION STATS
    # ========================================================

    def get_collection_stats(
        self,
    ) -> Dict[str, Any]:
        """Return collection statistics."""

        try:

            count = (
                self.collection.count()
            )

            filenames = (
                self.list_filenames()
            )

            return {
                "collection_name": (
                    self.collection_name
                ),
                "document_count": count,
                "unique_filenames": len(
                    filenames
                ),
                "filenames": filenames,
                "embedding_function": (
                    self.embedding_function_name
                ),
                "persist_directory": (
                    self.persist_directory
                ),
            }

        except Exception as exc:

            logger.error(
                "Failed to get collection stats: %s",
                exc,
            )

            return {}

    # ========================================================
    # METADATA PREPARATION
    # ========================================================

    def _prepare_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert metadata values to Chroma-compatible values."""

        prepared: Dict[str, Any] = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                prepared[str(key)] = str(
                    value
                )

            elif isinstance(
                value,
                (list, dict),
            ):
                prepared[str(key)] = str(
                    value
                )

            else:
                prepared[str(key)] = str(
                    value
                )

        return prepared

    # ========================================================
    # WHERE CLAUSE
    # ========================================================

    def _prepare_where_clause(
        self,
        filter_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert a simple filter dictionary into a Chroma where clause.

        Examples:

            {"filename": "notes.pdf"}

        becomes:

            {"filename": "notes.pdf"}

        Multiple values:

            {"filename": ["a.pdf", "b.pdf"]}

        becomes:

            {"filename": {"$in": ["a.pdf", "b.pdf"]}}
        """

        where_clause: Dict[str, Any] = {}

        for key, value in filter_dict.items():

            if value is None:
                continue

            key = str(key)

            if isinstance(value, str):
                where_clause[key] = value

            elif isinstance(value, list):

                clean_values = [
                    str(item)
                    for item in value
                    if item is not None
                ]

                if clean_values:
                    where_clause[key] = {
                        "$in": clean_values
                    }

            elif isinstance(
                value,
                bool,
            ):

                where_clause[key] = str(
                    value
                )

            else:

                where_clause[key] = str(
                    value
                )

        return where_clause

    # ========================================================
    # CLEAR COLLECTION
    # ========================================================

    def clear_collection(self) -> bool:
        """Delete every vector from the collection."""

        try:

            results = (
                self.collection.get(
                    include=[]
                )
            )

            ids = (
                results.get("ids") or []
            )

            if ids:
                self.collection.delete(
                    ids=ids
                )

            logger.info(
                "Cleared ChromaDB collection '%s'",
                self.collection_name,
            )

            return True

        except Exception as exc:

            logger.error(
                "Failed to clear collection: %s",
                exc,
            )

            return False

