import os

from tqdm import tqdm
from typing import List, Tuple

from llm_router_plugins.constants import _DontChangeMe

USE_LANGCHAIN_RAG = False

# The name of the FAISS collection to use (read from the environment).
LANGCHAIN_RAG_COLLECTION = os.getenv(
    f"{_DontChangeMe.MAIN_ENV_PREFIX}LANGCHAIN_RAG_COLLECTION", None
)

# Path or Hugging Face hub identifier of the embedding model.
LANGCHAIN_RAG_EMBEDDER = os.getenv(
    f"{_DontChangeMe.MAIN_ENV_PREFIX}LANGCHAIN_RAG_EMBEDDER", None
)

# Torch device for the model (e.g., "cpu" or "cuda:0").
LANGCHAIN_RAG_DEVICE = os.getenv(
    f"{_DontChangeMe.MAIN_ENV_PREFIX}LANGCHAIN_RAG_DEVICE", "cpu"
)

# Number of tokens per chunk (default: 400).
LANGCHAIN_RAG_CHUNK_SIZE = int(
    os.getenv(f"{_DontChangeMe.MAIN_ENV_PREFIX}LANGCHAIN_RAG_CHUNK_SIZE", "400")
)

# Number of tokens each chunk overlaps with the previous one (default: 100).
LANGCHAIN_RAG_CHUNK_OVERLAP = int(
    os.getenv(f"{_DontChangeMe.MAIN_ENV_PREFIX}LANGCHAIN_RAG_CHUNK_OVERLAP", "100")
)

# Store the FAISS index under the given directory
LANGCHAIN_RAG_PERSIST_DIR = os.getenv(
    f"{_DontChangeMe.MAIN_ENV_PREFIX}LANGCHAIN_RAG_PERSIST_DIR", None
)

if all(
    [
        LANGCHAIN_RAG_COLLECTION,
        LANGCHAIN_RAG_EMBEDDER,
        LANGCHAIN_RAG_DEVICE,
        LANGCHAIN_RAG_CHUNK_SIZE,
        LANGCHAIN_RAG_CHUNK_OVERLAP,
    ],
):
    USE_LANGCHAIN_RAG = True


if USE_LANGCHAIN_RAG:
    import torch
    from transformers import AutoTokenizer, AutoModel

    from langchain_core.documents import Document
    from langchain.embeddings.base import Embeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.faiss import DistanceStrategy


if USE_LANGCHAIN_RAG:

    class MeanPoolEmbeddings(Embeddings):
        """
        Implements the ``Embeddings`` interface required by LangChain
        (``embed_query`` and ``embed_documents``) using the same
        tokenizer/model you already have.
        """

        def __init__(self, tokenizer, model, device: str):
            self.tokenizer = tokenizer
            self.model = model
            self.device = device

        def embed_query(self, text: str) -> List[float]:
            """Embed a single query string."""
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, dim)
            mask = inputs.attention_mask.unsqueeze(-1)  # (1, seq_len, 1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, dim)
            return pooled.squeeze(0).cpu().tolist()

        def embed_documents(
            self, texts: List[str], batch_size: int = 32
        ) -> List[List[float]]:
            """Batch‑embed a list of documents using the specified batch size."""
            all_embeddings: List[List[float]] = []

            # Process the input list in batches, showing a progress bar.
            for start_idx in tqdm(
                range(0, len(texts), batch_size),
                desc="Embedding documents",
                unit="batch",
            ):
                batch = texts[start_idx : start_idx + batch_size]

                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                last_hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
                mask = inputs.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(
                    dim=1
                )  # (batch, dim)

                # Convert each vector in the batch to a plain Python list and extend the result.
                all_embeddings.extend([vec.cpu().tolist() for vec in pooled])

            return all_embeddings


class LangChainRAG:
    """
    A tiny RAG helper that:

    - creates a LangChain collection (FAISS + InMemoryDocstore)
      from a name passed to the ctor,
    - loads a Hugging Face embedding model from ``embedder_path`` onto ``device``,
    - splits incoming texts into token‑based chunks
      (window size and overlap are ctor arguments),
    - stores the chunks together with their embeddings in the collection,
    - provides a ``search`` method that returns the most similar chunks
      using cosine similarity.
    """

    def __init__(
        self,
        collection_name: str,
        embedder_path: str,
        device: str = "cpu",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        persist_dir: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        collection_name: str
            Identifier for the FAISS store – you can later
            retrieve it via ``FAISS.load_local``.
        embedder_path: str
            Path (or hub name) of a transformer that outputs sentence‑level
            embeddings (e.g. ``sentence-transformers/all-MiniLM-L6-v2``).
        device: str, default ``"cpu"``
            Torch device for the model (e.g. ``"cuda:0"``).
        chunk_size: int, default ``200``
            Number of tokens per chunk.
        chunk_overlap: int, default ``50``
            How many tokens the next chunk should overlap with the previous one.
        persist_dir: str | None, optional
            Directory where the FAISS index and docstore are saved.
            If ``None`` the ``collection_name`` is used as folder name.
        """
        if not USE_LANGCHAIN_RAG:
            raise Exception("Cannot use LangChainRAG when USE_LANGCHAIN_RAG=False!")

        self.persist_dir = persist_dir
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

        self.collection_name = collection_name
        self.doc_store = InMemoryDocstore()

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_path)
        self.model = AutoModel.from_pretrained(embedder_path).to(device)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedding = MeanPoolEmbeddings(
            tokenizer=self.tokenizer, model=self.model, device=self.device
        )
        self.vectorstore = self._prepare_faiss()

    # -------------------------------------------------------------------------
    def index_texts(self, texts: List[str]) -> None:
        """
        Split each text into token windows, embed each window,
        and push it to the FAISS store.
        """
        chunks, meta = self._split_into_chunks(texts)
        embeddings = self._embedding.embed_documents(chunks)
        docs = [
            Document(page_content=chunk, metadata=m)
            for chunk, m in zip(chunks, meta)
        ]
        self.vectorstore.add_embeddings(embeddings, docs)  # type: ignore
        self._persist()

    def search(self, text: str, top_n: int = 10) -> List["Document"]:
        """
        Retrieve the ``top_n`` most similar chunks to ``text`` from the FAISS index
        using cosine similarity.

        Parameters
        ----------
        text: str
            Query string.
        top_n: int, default ``10``
            Number of closest chunks to return.

        Returns
        -------
        List[Document]
            LangChain ``Document`` objects representing the most similar chunks,
            ordered from most to least similar.
        """
        if not getattr(self.vectorstore, "index", None):
            return []
        return self.vectorstore.similarity_search(text, k=top_n)

    # -------------------------------------------------------------------------
    def _persist(self) -> None:
        """
        Save the FAISS index and the in‑memory docstore to ``self.persist_dir``.
        """
        self.vectorstore.save_local(self.persist_dir)

    def _prepare_faiss(self):
        if self.persist_dir and os.listdir(self.persist_dir):
            return FAISS.load_local(
                folder_path=self.persist_dir,
                embeddings=self._embedding,
                docstore=self.doc_store,
                index_name="index",
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            return FAISS(
                embedding_function=self._embedding,
                docstore=self.doc_store,
                index=None,
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE,
            )

    def _split_into_chunks(self, texts: List[str]) -> Tuple[List[str], List[dict]]:
        """
        Token‑window split each text.

        Returns
        -------
        chunks: List[str]
            The raw string chunks.
        meta: List[dict]
            Metadata containing original text index and chunk index.
        """
        chunks, meta = [], []
        for doc_id, txt in enumerate(texts):
            token_ids = self.tokenizer.encode(txt, add_special_tokens=False)

            # slide a window over token ids
            for start in range(
                0, len(token_ids), self.chunk_size - self.chunk_overlap
            ):
                end = start + self.chunk_size
                window_ids = token_ids[start:end]
                if not window_ids:
                    continue
                chunk = self.tokenizer.decode(
                    window_ids, clean_up_tokenization_spaces=True
                )
                chunks.append(chunk)
                meta.append({"doc_id": doc_id, "chunk_id": len(meta)})
        return chunks, meta
