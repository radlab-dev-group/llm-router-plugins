#!/bin/bash

BASE_DIR=$(pwd)
MAIN_DATA_DIR="medusa_documentation"
FULL_DATA_DIR="${BASE_DIR}/${MAIN_DATA_DIR}"

function prepare_storage() {
    cd "$BASE_DIR" || { echo "❌ Cannot change to $BASE_DIR"; exit 1; }

    echo "🚀 Preparing storage in $BASE_DIR"

    # If the directory does NOT exist, create it (including parents if needed)
    if [[ ! -d "$MAIN_DATA_DIR" ]]; then
        echo "📁 Creating storage directory: $MAIN_DATA_DIR"
        mkdir -p "$MAIN_DATA_DIR"
        echo "✅ Directory created."
    else
        echo "ℹ️ Storage directory already exists: $MAIN_DATA_DIR"
    fi
}

function prepare_medusa_data() {
    # -------------------------------------------------
    echo "Preparing $FULL_DATA_DIR …"
    cd "$FULL_DATA_DIR" || { echo "❌ Cannot change to $FULL_DATA_DIR"; exit 1; }

    echo "🧹 Cleaning existing contents of $FULL_DATA_DIR"
    rm -rf "${FULL_DATA_DIR:?}"/*

    # -------------------------------------------------
    echo "📥 Downloading official Medusa docs"
    mkdir -p "$FULL_DATA_DIR/medusa-docs"
    cd "$FULL_DATA_DIR/medusa-docs" || { echo "❌ Cannot enter medusa-docs"; exit 1; }
    wget -q --show-progress https://docs.medusajs.com/llms-full.txt -O llms-full.txt
    echo "✅ Docs saved as llms-full.txt"

    # -------------------------------------------------
    echo "🔧 Cloning codee-sh extra plugins"
    cd "$FULL_DATA_DIR" || { echo "❌ Cannot return to $FULL_DATA_DIR"; exit 1; }
    mkdir -p codee-sh
    cd codee-sh || { echo "❌ Cannot enter codee-sh"; exit 1; }

    git clone https://github.com/codee-sh/medusa-plugin-automations
    git clone https://github.com/codee-sh/medusa-plugin-notification-emails

    echo "🎉 Medusa data preparation complete!"
}

#function index_medusa_data() {
#  export LLM_ROUTER_LANGCHAIN_RAG_COLLECTION=${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION:-"medusa_collection"}
#  export LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER=${LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER:-"/mnt/data2/llms/models/community/google/embeddinggemma-300m"}
#  export LLM_ROUTER_LANGCHAIN_RAG_DEVICE=${LLM_ROUTER_LANGCHAIN_RAG_DEVICE:-"cuda:2"}
#  export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE=${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE:-1024}
#  export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP=${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP:-100}
#  export LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR=${LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR:-"./workdir/plugins/utils/rag/langchain/${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}"}
#
#  llm-router-rag-langchain index --path "${FULL_DATA_DIR}" --ext .txt .md
#}


function index_medusa_data_to_kb() {
    # ----------------------------------------------
    export LLM_ROUTER_LANGCHAIN_RAG_COLLECTION=${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION:-"medusa_collection"}
    export LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER=${LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER:-"/mnt/data2/llms/models/community/google/embeddinggemma-300m"}
    export LLM_ROUTER_LANGCHAIN_RAG_DEVICE=${LLM_ROUTER_LANGCHAIN_RAG_DEVICE:-"cuda:2"}
    export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE=${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE:-1024}
    export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP=${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP:-100}
    export LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR=${LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR:-"./${FULL_DATA_DIR}/${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}"}

    # ----------------------------------------------
    if [[ -z "${FULL_DATA_DIR}" ]]; then
        echo "❌ FULL_DATA_DIR is not set. Please run prepare_medusa_data first."
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
        --ext .txt .md

    if [[ $? -eq 0 ]]; then
        echo "✅ Indexing completed successfully!"
    else
        echo "⚠️ Indexing finished with errors."
    fi
}

## 1. Prepare local dir
#prepare_storage
#
## 2. Download medusa documentation
#prepare_medusa_data

# 3. Prepare knowledge base
index_medusa_data_to_kb

cd "$BASE_DIR" || return
