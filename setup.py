import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
VERSION = (HERE / ".version").read_text(encoding="utf-8")

setup(
    name="radlab-llm-router-plugins",
    version=VERSION,
    description="Plugins for the LLM Router (guardrails, maskers, etc.)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="RadLab.dev Team",
    author_email="hello@radlab.dev",
    url="https://github.com/radlab-dev-group/llm-router-plugins",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "docs")),
    package_data={"llm_router_plugins.resources": ["*.json"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["radlab-pii-classification>=0.1.0"],
    extras_require={
        # Shared session memory for the Codex router (optional; routing stays
        # stateless without it).
        "memory": [
            "redis>=5",
        ],
        # Semantic routing (EmbeddingRouter): CPU build of FAISS.
        "ml": [
            "faiss-cpu",
            "sentence-transformers",
            "numpy",
            "scipy",
        ],
        # Same as "ml" but with the CUDA build of FAISS.
        "ml-gpu": [
            "faiss-gpu",
            "sentence-transformers",
            "numpy",
            "scipy",
        ],
        # LangChain-based RAG utilities.
        "rag": [
            "langchain",
            "Pillow",
        ],
        # Agentic knowledge layer: the SSE client and the stream/cache backend.
        # ``sse-lib`` is the Semantic Search Engine client; during development
        # install it from the sibling checkout instead
        # (``pip install ../semantic-search-engine/sse_lib``).
        "knowledge": [
            "sse-lib",
            "redis>=5",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-router-rag-langchain="
            "llm_router_plugins.cli.plugins.utils.rag.langchain:main",
            "llm-router-agentic-kb-worker="
            "llm_router_plugins.cli.plugins.utils.knowledge.worker:main",
            "llm-router-agentic-kb-docs="
            "llm_router_plugins.cli.plugins.utils.knowledge.docs:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
