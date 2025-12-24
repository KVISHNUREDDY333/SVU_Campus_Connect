# Campus Conversational Assistant (FastAPI + LangChain)

This project provides a FastAPI backend that ingests campus documents into MongoDB and a FAISS vectorstore and answers natural language queries using LangChain embeddings and (optionally) OpenAI LLM.

Quick start

1. Create a `.env` file (optional) to set `OPENAI_API_KEY` if you want generated answers:

```
OPENAI_API_KEY=sk-...
```

2. Start with Docker Compose (recommended):

```bash
docker-compose up --build
```

3. API endpoints
- `POST /ingest` - body: `{ "documents": [{"doc_id":"1","title":"IT Lab","content":"Room A101...","tags": ["lab"]}] }`
- `POST /query` - body: `{ "query": "Where is the IT lab?", "top_k": 4 }`

Notes
- Vector embeddings use `sentence-transformers/all-MiniLM-L6-v2` by default (no OpenAI key required for embeddings).
- If `OPENAI_API_KEY` is set, the app will use OpenAI via LangChain to generate a concise answer using retrieved documents.
# Campus_Connect_AI
