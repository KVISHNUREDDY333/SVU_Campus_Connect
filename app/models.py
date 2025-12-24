from pydantic import BaseModel
from typing import List, Optional


class DocumentItem(BaseModel):
    doc_id: str
    title: Optional[str] = None
    content: str
    tags: Optional[List[str]] = []


class IngestRequest(BaseModel):
    documents: List[DocumentItem]


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
