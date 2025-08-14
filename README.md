# RAG + HuggingFace LLM Question Answering Platform

A **Retrieval-Augmented Generation (RAG)** platform built with **FastAPI**, **FAISS**, and **SentenceTransformers**, capable of ingesting **PDF** documents and **JSON (Tarot)** datasets for semantic search and question answering.  
Supports HuggingFace LLM inference for generative answers, with extractive fallback when LLM is unavailable.

---

##  Features
- **PDF Ingestion & Semantic Search** – Extract text, split into overlapping chunks, and store in FAISS for fast retrieval.
- **JSON (Tarot) Ingestion** – Supports multiple tarot dataset formats for card meaning retrieval.
- **LLM Integration** – Uses HuggingFace Inference API (e.g., Mistral-7B) for answer generation.
- **Extractive Fallback** – If LLM is unavailable, returns top relevant sentences directly from source.
- **Citations in Answers** – Always includes 1–3 inline citations `(source|page#chunk)`.
- **User Feedback Logging** – Collects correct/wrong answers for model improvement.

---

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/Seleneplus/rag-llm-search.git
cd rag-llm-search
### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Set HuggingFace API key (optional, for LLM answers)

Get a free API token: [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)

```bash
# macOS/Linux
export HF_API_KEY="hf_xxx"

# Windows (PowerShell)
setx HF_API_KEY "hf_xxx"


