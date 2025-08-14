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

### 1. Installation
```bash
git clone https://github.com/Seleneplus/rag-llm-search.git
cd rag-llm-search
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set HuggingFace API key (optional, for LLM answers)

Get a free API token: HuggingFace Settings → Access Tokens
```bash
# macOS/Linux
export HF_API_KEY="hf_xxx"

# Windows (PowerShell)
setx HF_API_KEY "hf_xxx"
```

## 🚀 Running the App
```bash
uvicorn main:app --reload
```
Visit: http://localhost:8000

## 📌 API Endpoints
Health Check
```bash
GET /health
```
Response:
```bash
{"status": "ok"}
```


## 📌 Ingest PDF

```bash
POST /ingest/pdf
```
Content-Type: multipart/form-data

File: .pdf

Extracts text from PDF pages, splits into chunks, stores in FAISS.

Example Response:
```bash
{"status": "success", "pages_loaded": 10, "chunks_added": 85}
```

## 📌 Ingest JSON (Tarot)

```bash
POST /ingest/json

```
Content-Type: multipart/form-data

File: .json

Supports:

{"cards": [...]}

{"tarot_interpretations": [...]}

[ {...}, {...} ]

Example Response:
```bash
{"status": "success", "items_ingested": 156}
```

## Ask a Question
```bash
POST /ask
```
Request:
```bash
{
  "question": "What is supervised learning?",
  "top_k": 3,
  "min_score": 0.3
}
```
Response:
```bash
{
  "answer": "Supervised learning is a type of machine learning... (doc.pdf|p2#1)",
  "sources": ["doc.pdf|p2#1", "doc.pdf|p3#0"]
}
```

## Submit Feedback
``` bash
POST /feedback
```
Request:
```bash
{
  "question": "What is supervised learning?",
  "correct_answer": "It is a type of machine learning...",
  "wrong_answer": "I could not find the answer..."
}
```
Response:
```bash
{"status": "feedback stored"}
```

## 🖥 HTML UI
The root / endpoint serves a minimal UI:

1.Upload PDF

2.Ask a question

3.View answers with citations

## 📝 Notes
- For scanned PDFs (image-based), OCR preprocessing (e.g., pytesseract) is required before ingestion.
- Index and metadata are persisted in faiss_index.bin and metadata.json.
- By default, embeddings are generated using BAAI/bge-m3.

## 📄 License
MIT


