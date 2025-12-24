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

    import multiprocessing as _mp

    if _mp.get_start_method(allow_none=True) != "spawn":
        _mp.set_start_method("spawn", force=True)


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
        messages = None

        field_with_query = "user_last_statement"
        text_as_query = payload.get("user_last_statement")

        if not text_as_query:
            field_with_query = None

            messages = payload.get("messages")
            if messages:
                text_as_query = messages[-1].get("content")

        if not text_as_query:
            self._logger.error(f"Cannot find field with user text using {self.name}")
            return payload
        #
        # import json
        # print(json.dumps(payload, indent=2, ensure_ascii=False))

        docs = self.rag.search(text_as_query, top_n=50)
        for d in docs:
            self._logger.debug(d)

        extended_content = ""

        if messages:
            messages[-1]["content"] += extended_content
            payload["messages"] = messages
        elif field_with_query:
            payload[field_with_query] += extended_content

        return payload
