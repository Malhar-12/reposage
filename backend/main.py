import os
import json
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from rag import RAGSystem

app = FastAPI(title="RepoSage API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = RAGSystem()
indexing_status = {}


class IndexRequest(BaseModel):
    github_url: str

class ChatRequest(BaseModel):
    repo_id: str
    question: str
    history: list = []


async def _do_index(repo_id: str, github_url: str):
    try:
        logger.info(f"Indexing {repo_id}")
        indexing_status[repo_id] = {"status": "indexing", "message": "Fetching files from GitHub..."}
        result = await rag.index_repository(github_url)
        indexing_status[repo_id] = {
            "status": "ready",
            "message": f"Indexed {result['file_count']} files · {result['chunk_count']} chunks",
            **result,
        }
        logger.info(f"Done: {repo_id}")
    except Exception as e:
        logger.error(f"Failed {repo_id}: {e}")
        indexing_status[repo_id] = {"status": "error", "message": str(e)}


@app.post("/api/index")
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    try:
        owner, repo = rag._parse_url(req.github_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    repo_id = f"{owner}/{repo}"
    if repo_id in rag.collections:
        status = indexing_status.get(repo_id, {})
        return {"repo_id": repo_id, "status": "ready", **status}

    indexing_status[repo_id] = {"status": "indexing", "message": "Starting..."}
    background_tasks.add_task(_do_index, repo_id, req.github_url)
    return {"repo_id": repo_id, "status": "indexing"}


@app.get("/api/status/{repo_id:path}")
def get_status(repo_id: str):
    status = indexing_status.get(repo_id)
    if not status:
        if repo_id in rag.collections:
            return {"status": "ready", "message": "Already indexed"}
        return {"status": "not_found"}
    return status


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.repo_id not in rag.collections:
        raise HTTPException(status_code=404, detail="Repository not indexed yet.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def generate():
        try:
            async for chunk in rag.chat_stream(req.repo_id, req.question, req.history):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','content':str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/repos")
def list_repos():
    repos = []
    for repo_id, status in indexing_status.items():
        if status.get("status") == "ready":
            repos.append({
                "repo_id": repo_id,
                "file_count": status.get("file_count", 0),
                "chunk_count": status.get("chunk_count", 0),
                "summary": rag.summaries.get(repo_id, ""),
            })
    return {"repos": repos}


@app.get("/api/files/{repo_id:path}")
def get_files(repo_id: str):
    """Return file tree for the file explorer feature."""
    files = rag.file_lists.get(repo_id)
    if not files:
        raise HTTPException(status_code=404, detail="Repo not indexed.")
    return {"files": files}


@app.get("/api/summary/{repo_id:path}")
def get_summary(repo_id: str):
    """Return the auto-generated repo summary."""
    summary = rag.summaries.get(repo_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Repo not indexed.")
    return {"summary": summary}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "indexed_repos": len(rag.collections),
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "github_token_configured": bool(os.getenv("GITHUB_TOKEN")),
    }


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)