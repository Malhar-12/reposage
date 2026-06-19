import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from rag import RAGSystem

app = FastAPI(title="RepoSage API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGSystem()


class IndexRequest(BaseModel):
    github_url: str


class ChatRequest(BaseModel):
    repo_id: str
    question: str
    history: list = []


@app.post("/api/index")
async def index_repo(req: IndexRequest):
    try:
        result = await rag.index_repository(req.github_url)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.repo_id not in rag.collections:
        raise HTTPException(status_code=404, detail="Repository not indexed yet")

    async def generate():
        async for chunk in rag.chat_stream(req.repo_id, req.question, req.history):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/repos")
def list_repos():
    return {"repos": list(rag.collections.keys())}


@app.get("/health")
def health():
    return {"status": "ok", "indexed_repos": len(rag.collections)}


# Serve the frontend from the same origin (no CORS needed in production)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)