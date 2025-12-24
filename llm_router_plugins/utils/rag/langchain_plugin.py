import logging
from typing import Dict, Optional, Tuple, List

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.rag.engine.langchain import USE_LANGCHAIN_RAG

if USE_LANGCHAIN_RAG:
    from llm_router_plugins.utils.rag.engine.langchain import (
        LangChainRAG,
        LANGCHAIN_RAG_COLLECTION,
        LANGCHAIN_RAG_EMBEDDER,
        LANGCHAIN_RAG_DEVICE,
        LANGCHAIN_RAG_CHUNK_SIZE,
        LANGCHAIN_RAG_CHUNK_OVERLAP,
        LANGCHAIN_RAG_PERSIST_DIR,
    )


class LangchainRAGPlugin(PluginInterface):
    """
    Plugin that exposes a tiny Retrieval‑Augmented Generation (RAG) service
    built on LangChain + a FAISS (or Milvus) vector store.

    Expected ``payload`` dictionary:

    * ``action`` – either ``"index"`` or ``"search"``.
    * ``texts``  – list of strings (required for ``"index"``).
    * ``query``  – string (required for ``"search"``).
    * ``top_n``  – int, optional, defaults to 10 (used for ``"search"``).

    Returns a tuple ``(success, response_dict)`` where ``response_dict`` contains
    either ``"result"`` (list of documents) or ``"error"`` (error message).
    """

    name = "langchain_rag"

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise the plugin and the underlying ``LangChainRAG`` instance.

        Environment variables (all optional – sensible defaults are used when
        missing):

        * ``RAG_COLLECTION``       – name of the FAISS collection (default:
          ``"rag_collection"``)
        * ``RAG_EMBEDDER``        – Hugging‑Face model id/path
          (default: ``"sentence-transformers/all-MiniLM-L6-v2"``)
        * ``RAG_DEVICE``         – torch device (default: ``"cpu"``)
        * ``RAG_CHUNK_SIZE``      – token window size (default: ``200``)
        * ``RAG_CHUNK_OVERLAP``   – token overlap (default: ``50``)
        """
        super().__init__(logger=logger)

        if not USE_LANGCHAIN_RAG:
            raise Exception("Cannot use LangChainRAG when USE_LANGCHAIN_RAG=False!")

        self.rag = LangChainRAG(
            collection_name=LANGCHAIN_RAG_COLLECTION,
            embedder_path=LANGCHAIN_RAG_EMBEDDER,
            device=LANGCHAIN_RAG_DEVICE,
            chunk_size=LANGCHAIN_RAG_CHUNK_SIZE,
            chunk_overlap=LANGCHAIN_RAG_CHUNK_OVERLAP,
            persist_dir=LANGCHAIN_RAG_PERSIST_DIR,
        )

        # If a persisted FAISS index was loaded, consider the store ready.
        self._has_index = bool(getattr(self.rag.vectorstore, "index", None))

    def apply(self, payload: Dict) -> Dict:
        """
        Dispatch the incoming payload to the appropriate handler.

        The method extracts the ``action`` key and forwards the request to
        either :meth:`_index` or :meth:`_search`.  Any unexpected exception is
        caught, logged (if a logger is configured) and returned as a
        ``(False, {"error": …})`` tuple.

        Parameters
        ----------
        payload: dict
            Must contain an ``"action"`` entry – either ``"index"`` or
            ``"search"``.  The remaining keys are validated by the concrete
            handler.

        Returns
        -------
        Tuple[bool, dict]
            ``(True, {"result": …})`` on success, ``(False, {"error": …})`` on
            failure.
        """
        try:
            action = payload.get("action", "search")

            if action == "index":
                return self._index(payload)

            if action == "search":
                return self._search(payload)
            raise ValueError(f"Unsupported action '{action}'.")

        except Exception as exc:  # pragma: no cover – defensive logging
            self._logger.exception(f"LangchainRAGPlugin error {exc}")
            return False, {"error": str(exc)}

    def _index(self, payload: Dict) -> Tuple[bool, Dict]:
        """
        Handle the ``"index"`` action.

        This method validates that ``payload["texts"]`` is a list of strings,
        forwards the texts to the underlying :class:`LangChainRAG` instance,
        and records that an index has been created for the current session.

        Parameters
        ----------
        payload: dict
            Expected keys:
            * ``"texts"`` – ``List[str]`` containing the documents to index.

        Returns
        -------
        Tuple[bool, dict]
            ``(True, {"result": "..."} )`` on success.

        Raises
        ------
        ValueError
            If ``texts`` is missing or not a list of strings.
        """
        texts: List[str] = payload.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("'texts' must be a list of strings.")

        self.rag.index_texts(texts)
        self._has_index = True
        return True, {"result": f"Indexed {len(texts)} document(s)."}

    def _search(self, payload: Dict) -> Tuple[bool, Dict]:
        """
        Handle the ``"search"`` action.

        Ensures that an index exists, validates the query string, performs the
        similarity search using the configured ``top_n`` value and returns a
        JSON‑serialisable list of documents.

        Parameters
        ----------
        payload: dict
            Expected keys:
            * ``"query"`` – the search string.
            * ``"top_n"`` – optional ``int`` (default ``10``) specifying how many
              results to return.

        Returns
        -------
        Tuple[bool, dict]
            ``(True, {"result": [...]})`` where each entry contains ``content``
            and ``metadata`` fields.

        Raises
        ------
        RuntimeError
            If the vector store is empty (no prior indexing).
        ValueError
            If ``query`` is missing or not a string.
        """
        if not self._has_index:
            raise RuntimeError(
                "The vector store is empty – call the 'index' action first."
            )
        query: str = payload.get("query")
        if not isinstance(query, str):
            raise ValueError("'query' must be a string.")
        top_n: int = int(payload.get("top_n", 10))
        docs = self.rag.search(query, top_n=top_n)

        # Convert LangChain Document objects to a plain‑serialisable form
        result = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]
        return True, {"result": result}
