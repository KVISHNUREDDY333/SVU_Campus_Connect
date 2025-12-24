import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import IngestRequest, QueryRequest
from .db import kb_collection
from .vectorstore import get_vectorstore
from .config import settings
from .audio import router as audio_router

app = FastAPI(title="Campus Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vs = get_vectorstore()

# include audio router
app.include_router(audio_router, prefix="/audio")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    docs_to_add = []
    for d in req.documents:
        doc = d.dict()
        # store in mongodb
        kb_collection.update_one({"doc_id": doc["doc_id"]}, {"$set": doc}, upsert=True)
        docs_to_add.append({"content": doc["content"], "metadata": {"doc_id": doc["doc_id"], "title": doc.get("title"), "tags": doc.get("tags")}})
    try:
        vs.add_documents(docs_to_add)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vectorstore error: {e}")
    return {"ingested": len(docs_to_add)}


@app.post("/query")
def query(req: QueryRequest):
    query_text = req.query
    top_k = req.top_k or 4
    retrieved = vs.retrieve(query_text, k=top_k)
    snippets = []
    for r in retrieved:
        snippets.append({"content": r.page_content, "metadata": r.metadata})

    # If OpenAI key present, call generation; otherwise return retrieved snippets
    if settings.OPENAI_API_KEY:
        try:
            from langchain.llms import OpenAI
            prompt = """
You are a helpful campus assistant. Use the following documents to answer the final question. If the answer is not present, say you don't know.

Documents:
{docs}

Question: {query}
"""
            docs_text = "\n---\n".join([f"Title: {s['metadata'].get('title')}\n{s['content']}" for s in snippets])
            filled = prompt.format(docs=docs_text, query=query_text)
            llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY, temperature=0.2)
            answer = llm(filled)
            return {"answer": answer, "sources": snippets}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Fallback: return concatenated snippets and metadata
        combined = "\n\n".join([s["content"] for s in snippets])
        return {"answer": combined or "No documents found.", "sources": snippets}
