# Self-Hosted LLM Research Assistant Stack

This repository packages a production-ready, GPU-enabled research assistant stack composed of Ollama, ChromaDB, a Python RAG workbench, and LobeChat. The stack targets a single NVIDIA A40 (48 GB VRAM) and ships with automation to build a quantized Qwen2-72B model plus an embedding model for retrieval.

## Architecture

| Service    | Purpose | Key Features |
|------------|---------|--------------|
| **Ollama** | Hosts the Qwen2-72B (Q4_K_M) chat model and `multilingual-e5-large` embedding model with GPU acceleration. | Automatic GGUF download, custom Modelfile tuned for 48 GB VRAM, model pre-build on startup. |
| **ChromaDB** | Persistent vector database for document embeddings. | HTTP API exposed on port 8000, volume-backed storage. |
| **App (JupyterLab)** | Python RAG and evaluation workbench. | LangChain, LangGraph, Ragas, Whisper, yt-dlp preinstalled; exposes JupyterLab on port 8888. |
| **LobeChat** | Chat-first UI that speaks to Ollama via the internal Docker network. | Defaults to `qwen2-72b-q4km`, no external auth by default. |

All services communicate on an isolated Docker network (`llmnet`) and persist critical data through named Docker volumes.

## Prerequisites

- Docker Engine and Docker Compose Plugin (v2.20+ recommended) installed on the host.
- An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) configured. The stack expects at least one A40 (48 GB VRAM) or comparable GPU with compute capability.
- **Ensure the `QWEN72B_Q4KM_URL` environment variable in `docker-compose.yml` points to a publicly accessible Qwen2-72B Q4_K_M GGUF file**. The default is set to the Hugging Face download URL for `qwen2-72b-instruct-q4_k_m.gguf`, but you may mirror it locally for faster transfers. Even with 48 GB VRAM, keep the provided `num_ctx` and `num_batch` settings conservative unless you confirm additional headroom.

## Single-Command Setup

```bash
docker compose up -d --build
```

The command builds the custom Python app image, provisions volumes, pulls container images, launches the services, and prepares the Qwen2-72B Q4_K_M model automatically.

## How to Run

1. **Clone and configure**
   ```bash
   git clone https://github.com/your-org/llm-research-assistant-stack.git
   cd llm-research-assistant-stack
   # Edit docker-compose.yml to set a real QWEN72B_Q4KM_URL if you have not already.
   ```
2. **Launch the stack** (first run downloads the GGUF and builds the Ollama model):
   ```bash
   docker compose up -d --build
   ```
3. **Verify Ollama** is serving the custom model:
   ```bash
   curl http://localhost:11434/api/tags | jq
   # Look for "qwen2-72b-q4km"
   ```
4. **Access JupyterLab** at [http://localhost:8888](http://localhost:8888) (token is empty by default; set `JUPYTER_TOKEN` for security).
5. **Open LobeChat** at [http://localhost:3000](http://localhost:3000) and start chatting with the `qwen2-72b-q4km` model.
6. **Manage embeddings** via ChromaDB at `http://localhost:8000` (HTTP API) or by connecting through LangChain in notebooks.

## Service Configuration Details

- **Ollama** mounts two volumes: one for the Ollama store (`ollama_models`) and another (`./models`) for custom Modelfiles, weights, and initialization scripts. The container starts `ollama serve`, waits for the API, downloads the GGUF if necessary, builds the `qwen2-72b-q4km` model, and pre-pulls `multilingual-e5-large`.
- **ChromaDB** keeps vectors in the `chroma_data` volume. The default HTTP heartbeat endpoint (`/api/v1/heartbeat`) is used for health checks.
- **App** service exposes `/workspace/notebooks` and `/workspace/data` so you can sync notebooks and datasets from the host machine. It defaults to `DEFAULT_LLM_MODEL=qwen2-72b-q4km` for notebook utilities.
- **LobeChat** communicates with Ollama via the `ollama` hostname on the Docker network. Environment defaults select the custom model and disable external authentication.

## Testing and Evaluation

The following Jupyter-ready script demonstrates a full Retrieval-Augmented Generation (RAG) pipeline, including ingestion of a PDF and a YouTube transcript, embedding with `multilingual-e5-large`, querying the retriever, and evaluating the answer with Ragas. Paste the entire script into a single Jupyter notebook cell (or break into cells as desired). Update the `PDF_PATH` and `YOUTUBE_URL` variables with your own data before running.

```python
import os
from pathlib import Path
from typing import List

import chromadb

from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRelevancy

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_NAME = "research-assistant-demo"
DEFAULT_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "qwen2-72b-q4km")
EMBED_MODEL = "multilingual-e5-large"

PDF_PATH = Path("/workspace/data/sample.pdf")  # Change to your PDF path
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with a research video
QUESTION = "Summarize the key takeaways from the provided sources."

assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}. Upload a document to /workspace/data."

# ----------------------------------------------------------------------------
# Load source documents
# ----------------------------------------------------------------------------
pdf_loader = PyPDFLoader(str(PDF_PATH))
pdf_docs = pdf_loader.load()

yt_loader = YoutubeLoader.from_youtube_url(YOUTUBE_URL, add_video_info=True, language="en")
yt_docs = yt_loader.load()

all_docs = pdf_docs + yt_docs
print(f"Loaded {len(all_docs)} documents from PDF and YouTube transcript")

# ----------------------------------------------------------------------------
# Chunk and embed
# ----------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)
print(f"Created {len(chunks)} chunks")

embedding_fn = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

vectorstore.add_documents(chunks)
print("Chunks stored in ChromaDB")

# ----------------------------------------------------------------------------
# Run retrieval-augmented generation
# ----------------------------------------------------------------------------
llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

response = qa_chain.invoke({"query": QUESTION})
answer = response["result"]
contexts: List[str] = [doc.page_content for doc in response["source_documents"]]
print("Answer:\n", answer)

# ----------------------------------------------------------------------------
# Evaluate with Ragas
# ----------------------------------------------------------------------------
dataset = [
    {
        "question": QUESTION,
        "answer": answer,
        "contexts": contexts,
    }
]

metrics = [Faithfulness(), ContextRelevancy()]
report = evaluate(dataset, metrics)
print("Ragas evaluation:\n", report.to_pandas())
```

### LangGraph Agentic Workflow (Conceptual Example)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class ResearchState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve_node(state: ResearchState) -> ResearchState:
    """Call your retriever (e.g., Chroma + Ollama embeddings) and enrich context."""
    context = "...retrieved context..."
    return {**state, "context": context}

def synthesize_node(state: ResearchState) -> ResearchState:
    """Call the Ollama chat model to synthesize an answer."""
    answer = "...llm response using state['context']..."
    return {**state, "answer": answer}

workflow = StateGraph(ResearchState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()
result = app.invoke({"question": "What are the main findings?", "context": "", "answer": ""})
print(result["answer"])
```

## CI/CD Setup

Configure the following GitHub Secrets for the automated build and deployment workflows:

- **VULTR_HOST** → your server IP or DNS.
- **VULTR_USER** → SSH user (e.g., ubuntu).
- **VULTR_SSH_PRIVATE_KEY** → private key for that user (PEM contents).
- **VULTR_APP_PATH** → absolute path on server (e.g., /home/ubuntu/llm-stack).
- **GHCR_USER** and **GHCR_PAT** (only if the GHCR package is private). If public, these can be omitted.

## Shutdown and Cleanup

```bash
docker compose down
# To remove volumes (and models / embeddings) as well:
docker compose down -v
```

## Troubleshooting

- **Model download too slow**: Pre-download the GGUF file and host it on a fast server or LAN mirror, then update `QWEN72B_Q4KM_URL`.
- **Out-of-memory errors**: Reduce `num_ctx` or `num_batch` in `models/qwen2-72b-q4km/Modelfile` and rebuild by removing the Ollama model (`ollama rm qwen2-72b-q4km`) before restarting the stack.
- **Authentication**: Set `JUPYTER_TOKEN`, configure reverse proxies, or add VPN/VPC controls for production deployments.

Happy researching!
