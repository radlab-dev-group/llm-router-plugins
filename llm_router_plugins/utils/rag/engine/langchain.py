"""
CLI wrapper around the LangChainRAG helper.

Usage examples
--------------
# Index all *.txt and *.md files under ./data
python -m llm_router_plugins.utils.rag.engine.langchain_cli --index --path ./data --ext .txt .md

# Search the previously built index
python -m llm_router_plugins.utils.rag.engine.langchain_cli --search --query "What is LangChain?" --top_n 5
"""

import os
import re
import numpy as np

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
    import faiss
    import torch

    from transformers import AutoTokenizer, AutoModel

    from langchain_core.documents import Document
    from langchain.embeddings.base import Embeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.faiss import DistanceStrategy

    try:
        import torchvision

        torchvision.disable_beta_transforms_warning()
    except ImportError:
        torchvision = None
        pass


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

            # Normalize to unit length for cosine similarity
            vec = pooled.squeeze(0).cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec = vec / norm
            return vec.tolist()

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

                # ---- Normalise each vector in the batch ----
                batch_vecs = pooled.cpu().numpy()
                norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
                # Avoid division by zero
                norms[norms == 0] = 1
                batch_vecs = batch_vecs / norms

                # Convert each normalised vector to a plain Python list and extend the result.
                all_embeddings.extend([vec.tolist() for vec in batch_vecs])

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

    # -----------------------------------------------------------------
    def index_texts(self, texts: List[str], batch_size: int = 10) -> None:
        """
        Split each text into token windows, embed each window,
        and push it to the FAISS store.
        """
        texts = self._clear_texts(texts)
        chunks, meta = self._split_into_chunks(texts)
        docs = [
            Document(page_content=chunk, metadata=m)
            for chunk, m in zip(chunks, meta)
        ]

        for start_idx in range(0, len(docs), batch_size):
            batch = docs[start_idx : start_idx + batch_size]
            self.vectorstore.add_documents(batch)
        self._persist()

    def search(self, text: str, top_n: int = 10) -> List[Tuple["Document", float]]:
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
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search_with_score(text, k=top_n)

    # -----------------------------------------------------------------
    def _persist(self) -> None:
        """
        Save the FAISS index and the in‑memory docstore to ``self.persist_dir``.
        """
        if not self.persist_dir:
            return

        self.vectorstore.save_local(self.persist_dir)

    def _prepare_faiss(self):
        if self.persist_dir and os.listdir(self.persist_dir):
            return FAISS.load_local(
                folder_path=self.persist_dir,
                embeddings=self._embedding,
                index_name="index",
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
        else:
            # Ask the embedding object for the dimensionality of a single vector.
            dummy_vec = self._embedding.embed_query("dummy")
            dim = len(dummy_vec)

            # IndexFlatIP works with inner‑product; LangChain will normalize vectors
            # so this effectively gives you cosine similarity.
            faiss_index = faiss.IndexFlatIP(dim)

            return FAISS(
                embedding_function=self._embedding,
                docstore=self.doc_store,
                index=faiss_index,
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

    @staticmethod
    def _clear_texts(texts: List[str]) -> List[str]:
        """
        Strip leading/trailing whitespace and collapse any sequence of
        whitespace characters (spaces, tabs, newlines, etc.) to a single
        space. This makes the text uniform before further processing.
        """
        n_texts = []
        for text in texts:
            # Remove surrounding whitespace.
            text = text.strip()
            # Replace runs of whitespace with a single space.
            text = re.sub(r"\s+", " ", text)
            n_texts.append(text)
        return n_texts
