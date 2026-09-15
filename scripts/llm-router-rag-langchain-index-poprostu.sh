#!/bin/bash

BASE_DIR=$(pwd)
MAIN_DATA_DIR="ing_documentation"
FULL_DATA_DIR="/mnt/data2/dev/radlab-projekty/ING-Prostomat2/RAG/baza-PP-12.2025-RAG"

function index_data_to_kb() {
    # ----------------------------------------------
    export LLM_ROUTER_LANGCHAIN_RAG_COLLECTION=${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION:-${MAIN_DATA_DIR}}
    export LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER=${LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER:-"/mnt/data2/llms/models/community/google/embeddinggemma-300m"}
    export LLM_ROUTER_LANGCHAIN_RAG_DEVICE=${LLM_ROUTER_LANGCHAIN_RAG_DEVICE:-"cuda:2"}
    export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE=${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE:-1024}
    export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP=${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP:-100}
    export LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR=${LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR:-"./${FULL_DATA_DIR}/${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}"}

    # ----------------------------------------------
    if [[ -z "${FULL_DATA_DIR}" ]]; then
        echo "❌ FULL_DATA_DIR is not set. Please prepare data first."
        return 1
    fi

    if [[ ! -d "${FULL_DATA_DIR}" ]]; then
        echo "❌ Data directory '${FULL_DATA_DIR}' does not exist. Nothing to index."
        return 1
    fi

    # ----------------------------------------------
    echo "🚀 Starting RAG indexing for collection '${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}'"
    llm-router-rag-langchain index \
        --path "${FULL_DATA_DIR}" \
        --ext .txt .md .html

    if [[ $? -eq 0 ]]; then
        echo "✅ Indexing completed successfully!"
    else
        echo "⚠️ Indexing finished with errors."
    fi
}

index_data_to_kb

cd "$BASE_DIR" || return
