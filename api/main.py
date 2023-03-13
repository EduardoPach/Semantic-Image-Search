from __future__ import annotations

import io
from typing import Union, List

import faiss
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, conlist

from api import SemanticSearcher

class Query(BaseModel):
    query: conlist(Union[str, bytes], min_items=1, max_items=5)
    k: int = 10

class SearchResult(BaseModel):
    urls: conlist(List[str], min_items=1, max_items=5)

app = FastAPI(title="Image Semantic Search - Unsplash")


@app.on_event("startup")
def load_searcher() -> None:
    index = faiss.read_index("embeddings/00001.index")
    db = pd.read_csv("embeddings/00001.csv")
    global searcher
    searcher = SemanticSearcher("openai/clip-vit-base-patch32", index, db)


@app.get("/")
def home() -> None:
    return "Welcome to the Image Semantic Search API. Head over http://localhost:8000/docs for more info."

@app.post("/search", response_model=SearchResult)
def search(query_batch: Query) -> SearchResult:
    query = query_batch.query
    k = query_batch.k
    if not isinstance(query, list):
        HTTPException(status_code=400, detail="Query must be a list")
    elif isinstance(query[0], bytes):
        query = [Image.open(io.BytesIO(q)) for q in query]
    elif not isinstance(query[0], str):
        HTTPException(status_code=400, detail="Query must be a list of strings or image bytes")
    urls = searcher(query, k)
    return SearchResult(urls=urls)
